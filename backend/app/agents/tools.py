"""Tool Registry (Phase 4 roadmap, Section 17-18: "Tool Registry + tool safety").

A small registry of REAL callable tools agents can invoke, each tagged with
a permission level. call_tool() is the single safety gate every tool call
goes through:

  - unknown tool name -> rejected
  - WRITE/DANGEROUS tools -> rejected unless explicitly allow-listed (nothing
    is allow-listed yet - every tool below is READ_ONLY, so the check is
    enforced but not yet exercised; it's the hook a future write-capable
    tool - e.g. "apply_cleaning" - plugs into)
  - a per-workflow call budget, so a runaway agent can't spam a tool
  - every call (success or failure) is written into state.tool_results and
    state.traces, so the full tool-use history is inspectable, not just
    hidden inside a model's reasoning

Currently used by agents/security.py: the Security agent's PII check is no
longer just "ask the model to eyeball it" - pii_scan() is a deterministic,
repeatable regex check layered on top of the LLM's own judgment.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from app.agents.state import WorkflowState
from app.agents.tracing import add_trace


class ToolPermission(str, Enum):
    READ_ONLY = "READ_ONLY"
    WRITE = "WRITE"
    DANGEROUS = "DANGEROUS"


class ToolCallError(Exception):
    pass


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    permission: ToolPermission = ToolPermission.READ_ONLY
    max_calls_per_workflow: int = 20


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def _pii_scan(text: str) -> dict:
    """Deterministic regex scan for common PII patterns. A heuristic, not a
    guarantee - it deliberately errs toward flagging too much (a phone-shaped
    number could just be a stat) since a false positive costs a review, a
    false negative could leak real PII into a user-facing report."""
    findings = []
    for label, pattern in _PII_PATTERNS.items():
        matches = pattern.findall(text or "")
        if matches:
            findings.append({"type": label, "count": len(matches)})
    return {"clean": len(findings) == 0, "findings": findings}


_INJECTION_PATTERNS = {
    "instruction_override": re.compile(
        r"\b(ignore|disregard|forget)\s+(all\s+)?(the\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?)\b",
        re.IGNORECASE,
    ),
    "role_override": re.compile(
        r"\b(you are now|act as|pretend (to be|you are)|roleplay as|from now on you)\b",
        re.IGNORECASE,
    ),
    "prompt_exfiltration": re.compile(
        r"\b(reveal|show|print|output|repeat)\s+(me\s+)?your\s+(system\s+)?(prompt|instructions)\b",
        re.IGNORECASE,
    ),
    "sql_injection_shape": re.compile(
        r"(;\s*drop\s+table|'\s*or\s*'1'\s*=\s*'1|--\s*$|;\s*delete\s+from)",
        re.IGNORECASE,
    ),
    "safety_bypass": re.compile(
        r"\b(bypass|disable|turn off)\s+(the\s+)?(safety|security|filter|guardrail)s?\b",
        re.IGNORECASE,
    ),
}


def _prompt_injection_scan(text: str) -> dict:
    """Deterministic keyword/pattern scan for common prompt-injection and
    jailbreak framings in a user's free-text query. Same philosophy as
    _pii_scan: a heuristic that errs toward flagging too much - this is a
    data analysis platform, not a general chatbot, so legitimate queries
    essentially never need phrasing like "ignore your previous
    instructions" or "reveal your system prompt"."""
    findings = []
    for label, pattern in _INJECTION_PATTERNS.items():
        if pattern.search(text or ""):
            findings.append({"type": label})
    return {"clean": len(findings) == 0, "findings": findings}


def _get_statistics(dataset_id: str, user_id: str) -> dict:
    from app.services.datasets import build_data_summary, get_dataset_dataframe

    dataframe = get_dataset_dataframe(dataset_id, user_id)
    return {"data_summary": build_data_summary(dataframe)}


def _load_dataset_sample(dataset_id: str, user_id: str, n: int = 5) -> dict:
    from app.services.datasets import get_dataset_dataframe

    dataframe = get_dataset_dataframe(dataset_id, user_id)
    return {"sample": dataframe.head(n).to_dict(orient="records")}


_SQL_FORBIDDEN_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex|replace|exec)\b",
    re.IGNORECASE,
)


def _validate_sql(sql_query: str) -> dict:
    """Deterministic guard: only a single, plain SELECT statement is
    allowed. This is the hard backstop every SQL execution goes through
    regardless of what the LLM generated - the same "never trust the
    model's output alone" philosophy as pii_scan/prompt_injection_scan."""
    query = (sql_query or "").strip()
    if not query:
        return {"valid": False, "reason": "Empty query"}
    if ";" in query.rstrip(";"):
        return {"valid": False, "reason": "Multiple statements are not allowed"}
    if not query.lower().lstrip().startswith("select"):
        return {"valid": False, "reason": "Only SELECT statements are allowed"}
    if _SQL_FORBIDDEN_KEYWORDS.search(query):
        return {"valid": False, "reason": "Query contains a forbidden keyword - only read-only SELECT is allowed"}
    return {"valid": True, "reason": ""}


def _inspect_schema(dataset_id: str, user_id: str) -> dict:
    from app.services.datasets import get_dataset_dataframe

    dataframe = get_dataset_dataframe(dataset_id, user_id)
    columns = [{"name": col, "type": str(dtype)} for col, dtype in dataframe.dtypes.items()]
    return {"table_name": "data", "columns": columns, "row_count": len(dataframe)}


def _execute_read_only_query(dataset_id: str, user_id: str, sql_query: str, max_rows: int = 100) -> dict:
    """Executes a validated SELECT-only query against the dataset, loaded
    into a throwaway in-memory SQLite database (stdlib `sqlite3` - no new
    dependency, no persistent database ever touched) as a table named
    `data`. The connection is created and discarded within this single
    call - nothing about this dataset's real storage (Supabase) is
    reachable from the query at all, by construction, not just by policy."""
    import sqlite3

    from app.services.datasets import get_dataset_dataframe

    validation = _validate_sql(sql_query)
    if not validation["valid"]:
        return {"error": f"SQL rejected: {validation['reason']}"}

    dataframe = get_dataset_dataframe(dataset_id, user_id)
    conn = sqlite3.connect(":memory:")
    try:
        dataframe.to_sql("data", conn, index=False, if_exists="replace")
        cursor = conn.execute(sql_query)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(max_rows)
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": len(rows) == max_rows,
        }
    except sqlite3.Error as e:
        return {"error": f"SQL execution failed: {e}"}
    finally:
        conn.close()


TOOL_REGISTRY: dict[str, Tool] = {
    "pii_scan": Tool(
        name="pii_scan",
        description="Scans a block of text for likely PII (emails, phone numbers, SSN-shaped numbers).",
        fn=_pii_scan,
        permission=ToolPermission.READ_ONLY,
    ),
    "prompt_injection_scan": Tool(
        name="prompt_injection_scan",
        description="Scans free-text (e.g. a user query) for common prompt-injection/jailbreak patterns.",
        fn=_prompt_injection_scan,
        permission=ToolPermission.READ_ONLY,
    ),
    "get_statistics": Tool(
        name="get_statistics",
        description="Recomputes the privacy-filtered statistical summary for a dataset.",
        fn=_get_statistics,
        permission=ToolPermission.READ_ONLY,
    ),
    "load_dataset_sample": Tool(
        name="load_dataset_sample",
        description="Loads the first N rows of a dataset as records (small samples only).",
        fn=_load_dataset_sample,
        permission=ToolPermission.READ_ONLY,
    ),
    "inspect_schema": Tool(
        name="inspect_schema",
        description="Returns the dataset's column names and types, as if it were a SQL table named `data`.",
        fn=_inspect_schema,
        permission=ToolPermission.READ_ONLY,
    ),
    "validate_sql": Tool(
        name="validate_sql",
        description="Checks a SQL query is a single, safe, read-only SELECT statement before it's ever run.",
        fn=_validate_sql,
        permission=ToolPermission.READ_ONLY,
    ),
    "execute_read_only_query": Tool(
        name="execute_read_only_query",
        description=(
            "Runs a validated SELECT query against the dataset in a throwaway in-memory "
            "database (table name `data`). Rejects anything that isn't a single SELECT."
        ),
        fn=_execute_read_only_query,
        permission=ToolPermission.READ_ONLY,
    ),
}

# In-memory per-workflow call counters. Fine for a single-process deployment;
# would move to Redis/Supabase if this ever runs multi-process/multi-worker.
_call_counts: dict[tuple[str, str], int] = {}


def call_tool(
    name: str,
    *,
    requesting_agent: str,
    state: WorkflowState | None = None,
    **kwargs: Any,
) -> dict:
    """The single safety gate every tool call goes through. Returns the
    tool's result dict on success; raises ToolCallError on any safety
    rejection or execution failure - callers decide whether that's fatal
    for their step."""
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        raise ToolCallError(f"Unknown tool: {name}")

    if tool.permission != ToolPermission.READ_ONLY:
        raise ToolCallError(
            f"Tool '{name}' requires permission {tool.permission.value}, "
            f"which is not yet granted to any agent."
        )

    workflow_id = state.workflow_id if state else "no-workflow"
    key = (workflow_id, name)
    _call_counts[key] = _call_counts.get(key, 0) + 1
    if _call_counts[key] > tool.max_calls_per_workflow:
        raise ToolCallError(
            f"Tool '{name}' exceeded its call budget ({tool.max_calls_per_workflow}/workflow)."
        )

    started = time.time()
    try:
        result = tool.fn(**kwargs)
        if state is not None:
            state.tool_results[f"{name}:{started}"] = {
                "tool": name,
                "requesting_agent": requesting_agent,
                "args": {k: v for k, v in kwargs.items() if k != "text"},  # don't dump raw text into state
                "result": result,
            }
            add_trace(state, requesting_agent, f"tool_call:{name}", input_data=kwargs, output_data=result)
        return result
    except Exception as e:
        if state is not None:
            add_trace(state, requesting_agent, f"tool_call:{name}", input_data=kwargs, error=str(e))
        raise ToolCallError(f"Tool '{name}' failed: {e}") from e


def list_tools() -> list[dict]:
    """Registry snapshot for the Manager's prompt context / a future
    /agent/tools endpoint."""
    return [
        {"name": t.name, "description": t.description, "permission": t.permission.value}
        for t in TOOL_REGISTRY.values()
    ]
