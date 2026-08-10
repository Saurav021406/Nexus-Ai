from app.agents import education, finance, generic, healthcare, hr, retail

AGENT_REGISTRY = {
    "Education": education.analyze,
    "Retail": retail.analyze,
    "Finance": finance.analyze,
    "HR": hr.analyze,
    "Healthcare": healthcare.analyze,
    "General": generic.analyze,
}


def get_agent(domain: str):
    return AGENT_REGISTRY.get(domain)


def available_agent_domains() -> set[str]:
    return set(AGENT_REGISTRY.keys())
