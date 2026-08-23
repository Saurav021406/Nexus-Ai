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

from app.agents import business_analyst, data_engineer, data_scientist, education, finance, generic, healthcare, hr, ml_engineer, research_agent, retail, sql_agent, visualization_agent
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
    # Most agents only ever see the privacy-filtered data_summary text - that's
    # enough for reasoning-over-a-summary agents. The SQL Agent genuinely needs
    # dataset_id/user_id to run real queries, so it opts into getting the full
    # WorkflowState instead - see manager_v2.run_specialist_task for the branch.
    needs_full_access: bool = False

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
    # --- Functional agents (Section 16) - domain-agnostic, chosen by WHAT
    # kind of analysis is needed rather than WHAT industry the data is from.
    # The Manager can combine these with a domain specialist in one plan,
    # e.g. Finance + Business Analyst for a finance question that also
    # needs an executive-summary framing. ---
    "Data Engineer": AgentDefinition(
        name="Data Engineer",
        description="Assesses data quality: missing values, outliers, type issues, cleanliness.",
        fn=data_engineer.analyze,
        capabilities=["data_quality_assessment", "cleaning_recommendations"],
        tools=["get_statistics", "load_dataset_sample"],
    ),
    "Data Scientist": AgentDefinition(
        name="Data Scientist",
        description="Statistical analysis: correlations, distributions, trends, testable hypotheses.",
        fn=data_scientist.analyze,
        capabilities=["statistics", "eda", "correlation_analysis", "distribution_analysis"],
        tools=["get_statistics", "load_dataset_sample"],
    ),
    "ML Engineer": AgentDefinition(
        name="ML Engineer",
        description=(
            "Assesses modeling/forecasting readiness and recommends an approach. "
            "Does not train models - orchestration-layer reasoning only (Phase 4 scope)."
        ),
        fn=ml_engineer.analyze,
        capabilities=["modeling_readiness_assessment", "target_variable_identification"],
        tools=["get_statistics", "load_dataset_sample"],
    ),
    "Business Analyst": AgentDefinition(
        name="Business Analyst",
        description="Translates technical findings into business meaning, KPIs, and recommendations.",
        fn=business_analyst.analyze,
        capabilities=["business_interpretation", "kpi_analysis", "executive_summary"],
        tools=["get_statistics", "load_dataset_sample"],
    ),
    "Visualization": AgentDefinition(
        name="Visualization",
        description="Recommends which chart type(s) best represent the data. Does not render charts.",
        fn=visualization_agent.analyze,
        capabilities=["chart_type_recommendation"],
        tools=["get_statistics", "load_dataset_sample"],
    ),
    "SQL": AgentDefinition(
        name="SQL",
        description=(
            "Answers precise, row-level questions by writing and safely running a real "
            "read-only SQL query against the dataset (treated as a single table `data`)."
        ),
        fn=sql_agent.analyze,
        capabilities=["sql_generation", "sql_validation", "read_only_query_execution"],
        tools=["inspect_schema", "validate_sql", "execute_read_only_query"],
        input_schema="full WorkflowState (needs dataset_id/user_id, not just data_summary)",
        output_schema="{summary, key_metrics, recommendation, sql_query, columns, row_count}",
        needs_full_access=True,
    ),
    "Research": AgentDefinition(
        name="Research",
        description=(
            "Answers research/background-context questions using general knowledge. "
            "NOT grounded in any retrieved document or source yet - that's Phase 6 RAG. "
            "Every answer is explicitly flagged as ungrounded (grounded=False)."
        ),
        fn=research_agent.analyze,
        capabilities=["general_knowledge_research"],
        tools=["get_statistics", "load_dataset_sample"],
        output_schema="{summary, key_metrics, recommendation, grounded}",
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

