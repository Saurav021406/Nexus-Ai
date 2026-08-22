"""SQL Agent (Section 16 of the Phase 4 spec).

"Phase 4 should provide the agent interface and tools... Do not implement
the full Phase 7 live database system here." There's no live SQL database
backing datasets in this app (they're CSV/Excel uploads), so this agent
treats the uploaded dataset as a single-table database (table `data`) and
answers questions by:
  1. inspect_schema - see the real columns/types
  2. writing one SELECT query for the question
  3. validate_sql - deterministic safety check (see agents/tools.py)
  4. execute_read_only_query - runs it against a throwaway in-memory
     SQLite database built from the dataframe, never touching Supabase

This is genuinely real SQL execution, not a simulation - but scoped to
read-only queries against one dataset, which is the Phase 4 boundary.

Unlike every other agent here, this one needs dataset_id/user_id (not just
the privacy-filtered data_summary text), so it's registered with
needs_full_access=True and receives the full WorkflowState - see
agents/manager_v2.py's run_specialist_task for the branch that makes that
call.
"""

from app.agents.state import WorkflowState
from app.agents.tools import ToolCallError, call_tool
from app.services.consensus import get_consensus_json


def analyze(state: WorkflowState, task_description: str = "") -> dict:
    task_description = task_description or state.user_query or "Summarize this dataset"

    try:
        schema = call_tool(
            "inspect_schema", requesting_agent="SQL", state=state,
            dataset_id=state.dataset_id, user_id=state.user_id,
        )
    except ToolCallError as e:
        return {"error": f"Could not inspect schema: {e}"}

    prompt = f"""You are a SQL Agent. The dataset is available as a single table named
`data` with this exact schema:

{schema['columns']}
(row_count: {schema['row_count']})

Write ONE SQL SELECT statement (SQLite syntax) that answers this request:
{task_description}

Rules:
- Only a single SELECT statement - no other statement type, no semicolons.
- Only use column names that actually appear in the schema above.
- Keep it focused and add LIMIT if the result could be large.

Respond ONLY with this exact JSON shape (no markdown, no extra text):
{{"sql_query": "SELECT ..."}}"""

    try:
        plan = get_consensus_json(prompt, temperature=0.3, max_tokens=512)
    except Exception as e:
        return {"error": f"Could not generate SQL: {e}"}

    sql_query = (plan.get("sql_query") or "").strip()
    if not sql_query:
        return {"error": "Generated no SQL query"}

    try:
        query_result = call_tool(
            "execute_read_only_query", requesting_agent="SQL", state=state,
            dataset_id=state.dataset_id, user_id=state.user_id, sql_query=sql_query,
        )
    except ToolCallError as e:
        return {"error": f"Query execution blocked: {e}"}

    if "error" in query_result:
        return {"error": query_result["error"], "sql_query": sql_query}

    rows = query_result["rows"]
    columns = query_result["columns"]
    preview_rows = rows[:5]

    key_metrics = [
        ", ".join(f"{col}={val}" for col, val in zip(columns, row)) for row in preview_rows
    ]

    summary = (
        f"Ran `{sql_query}` against the dataset - returned {query_result['row_count']} row(s)"
        f"{' (truncated to first 100)' if query_result.get('truncated') else ''}."
    )

    return {
        "summary": summary,
        "key_metrics": key_metrics or ["Query returned no rows"],
        "recommendation": "Refine the query with additional filters if this isn't specific enough.",
        "sql_query": sql_query,
        "columns": columns,
        "row_count": query_result["row_count"],
    }
