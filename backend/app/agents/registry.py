"""Agent Registry (Section 15 of the Phase 4 spec).

Upgrades the old flat {domain: function} map into structured
AgentDefinitions with capabilities, tools, and permissions, so the new
intent-aware Manager (agents/manager_v2.py) can describe what's available
to the LLM instead of relying on hard-coded domain routing.

Backward compatible: get_agent() and available_agent_domains() are the same
functions the old domain-routing Manager (agents/manager.py) already calls -
their behavior is unchanged, they just now read from AGENT_DEFINITIONS
instead of a plain dict literal.
"""

from dataclasses import dataclass, field
from typing import Callable

from app.agents import education, finance, generic, healthcare, hr, retail
from app.agents.tools import list_tools


@dataclass
class AgentDefinition:
    name: str
    description: str
    fn: Callable
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    input_schema: str = "data_summary: str (privacy-filtered dataset profile)"
    output_schema: str = "{summary, key_metrics, recommendation}"
    permissions: str = "READ_ONLY"
    status: str = "active"
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "permissions": self.permissions,
            "status": self.status,
            "version": self.version,
        }


AGENT_DEFINITIONS: dict[str, AgentDefinition] = {
    "Education": AgentDefinition(
        name="Education",
        description="Domain specialist for education/academic datasets (grades, attendance, courses).",
        fn=education.analyze,
        capabilities=["academic_performance_analysis", "engagement_trends"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
    "Retail": AgentDefinition(
        name="Retail",
        description="Domain specialist for retail/e-commerce datasets (sales, inventory, orders).",
        fn=retail.analyze,
        capabilities=["sales_trend_analysis", "inventory_insight", "customer_behavior"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
    "Finance": AgentDefinition(
        name="Finance",
        description="Domain specialist for financial datasets (revenue, expenses, transactions).",
        fn=finance.analyze,
        capabilities=["financial_trend_analysis", "risk_indicators"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
    "HR": AgentDefinition(
        name="HR",
        description="Domain specialist for HR/people datasets (headcount, attrition, performance).",
        fn=hr.analyze,
        capabilities=["attrition_analysis", "workforce_insight"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
    "Healthcare": AgentDefinition(
        name="Healthcare",
        description="Domain specialist for healthcare/clinical datasets.",
        fn=healthcare.analyze,
        capabilities=["clinical_trend_analysis", "patient_outcome_insight"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
    "General": AgentDefinition(
        name="General",
        description="Fallback specialist for datasets that don't fit a specific domain.",
        fn=generic.analyze,
        capabilities=["general_statistical_analysis"],
        tools=["get_statistics", "load_dataset_sample"],  # real Tool Registry names, see agents/tools.py
    ),
}


def get_agent(domain: str) -> Callable | None:
    definition = AGENT_DEFINITIONS.get(domain)
    return definition.fn if definition else None


def get_agent_definition(domain: str) -> AgentDefinition | None:
    return AGENT_DEFINITIONS.get(domain)


def available_agent_domains() -> set[str]:
    return set(AGENT_DEFINITIONS.keys())


def list_capabilities() -> list[dict]:
    """Full registry snapshot, in the shape the Manager Agent's prompt
    context and the future /agent/registry endpoint both want."""
    return [d.to_dict() for d in AGENT_DEFINITIONS.values()]


# Fail fast on a typo'd tool name rather than silently telling the Manager
# an agent has a tool that doesn't exist in the real Tool Registry.
_known_tool_names = {t["name"] for t in list_tools()}
for _definition in AGENT_DEFINITIONS.values():
    _unknown = set(_definition.tools) - _known_tool_names
    if _unknown:
        raise RuntimeError(
            f"Agent '{_definition.name}' references unknown tool(s) {_unknown} "
            f"- not present in app.agents.tools.TOOL_REGISTRY."
        )

