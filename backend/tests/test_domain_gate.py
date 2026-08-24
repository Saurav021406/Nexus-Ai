from app.agents.domain_gate import check_domain_relevance

COLUMNS = ["region", "revenue", "product_category", "order_date"]
SUMMARY = "mean revenue: 1500, top region: North, total orders: 342"


def test_query_overlapping_dataset_columns_is_in_domain():
    result = check_domain_relevance("Which product_category sells best?", COLUMNS, SUMMARY)
    assert result["in_domain"] is True


def test_pure_vocabulary_overlap_without_generic_words():
    columns = ["churn_risk_score", "subscription_tier"]
    result = check_domain_relevance("Which subscription tier has highest churn risk?", columns, "")
    assert result["in_domain"] is True
    assert "overlaps dataset vocabulary" in result["reason"]


def test_generic_analysis_language_is_always_in_domain():
    result = check_domain_relevance("Can you summarize this and give me insights?", COLUMNS, SUMMARY)
    assert result["in_domain"] is True


def test_clearly_off_topic_query_is_rejected():
    result = check_domain_relevance("What is the capital of France?", COLUMNS, SUMMARY)
    assert result["in_domain"] is False


def test_unrelated_creative_request_is_rejected():
    result = check_domain_relevance("Write me a poem about the ocean", COLUMNS, SUMMARY)
    assert result["in_domain"] is False


def test_very_short_query_defaults_to_allow():
    result = check_domain_relevance("hi", COLUMNS, SUMMARY)
    assert result["in_domain"] is True


def test_empty_query_defaults_to_allow():
    result = check_domain_relevance("", COLUMNS, SUMMARY)
    assert result["in_domain"] is True
