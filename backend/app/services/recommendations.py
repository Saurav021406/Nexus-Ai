"""Interaction-based product recommendations from an uploaded dataset."""

from __future__ import annotations

import math

import pandas as pd


def recommend_products(
    dataframe: pd.DataFrame,
    *,
    user_column: str,
    user_value: str,
    item_column: str,
    rating_column: str | None = None,
    limit: int = 10,
) -> dict:
    if user_column not in dataframe.columns or item_column not in dataframe.columns:
        raise ValueError("Select valid customer and product columns.")
    if user_column == item_column:
        raise ValueError("Customer and product columns must be different.")
    if rating_column and rating_column not in dataframe.columns:
        raise ValueError("Select a valid rating column.")
    if rating_column and rating_column in {user_column, item_column}:
        raise ValueError("The rating column must be different from the customer and product columns.")
    if dataframe.empty:
        raise ValueError("The dataset contains no interactions.")

    columns = [user_column, item_column] + ([rating_column] if rating_column else [])
    interactions = dataframe[columns].dropna(subset=[user_column, item_column]).copy()
    interactions[user_column] = interactions[user_column].astype(str)
    interactions[item_column] = interactions[item_column].astype(str)
    interactions = interactions[
        (interactions[user_column].str.len() > 0) & (interactions[item_column].str.len() > 0)
    ]

    customer_key = str(user_value).strip()
    if not customer_key:
        raise ValueError("Enter a customer value.")

    if rating_column:
        interactions[rating_column] = pd.to_numeric(interactions[rating_column], errors="coerce")
        interactions = interactions.dropna(subset=[rating_column])
        if interactions.empty:
            raise ValueError("The selected rating column has no numeric values.")
        interactions = (
            interactions.groupby([user_column, item_column], as_index=False)[rating_column].mean()
        )
        customer_history = interactions[interactions[user_column] == customer_key]
        if customer_history.empty:
            raise ValueError(f"No interactions were found for customer '{customer_key}'.")
        global_min = float(interactions[rating_column].min())
        global_max = float(interactions[rating_column].max())
        rating_range = global_max - global_min
        history_weights = {
            row[item_column]: (
                max(0.0, float(row[rating_column]) - global_min) / rating_range
                if rating_range > 0
                else 1.0
            )
            for _, row in customer_history.iterrows()
        }
    else:
        interactions = interactions.drop_duplicates([user_column, item_column])
        customer_history = interactions[interactions[user_column] == customer_key]
        if customer_history.empty:
            raise ValueError(f"No interactions were found for customer '{customer_key}'.")
        history_weights = {item: 1.0 for item in customer_history[item_column]}

    if not any(weight > 0 for weight in history_weights.values()):
        raise ValueError(
            "This customer has no positive ratings to base recommendations on. "
            "Choose a rating column with meaningful positive scores or omit it."
        )

    item_users = {
        item: set(group[user_column])
        for item, group in interactions.groupby(item_column, sort=False)
    }
    seen_items = set(history_weights)
    scores: dict[str, float] = {}
    evidence: dict[str, int] = {}

    for history_item, weight in history_weights.items():
        if weight <= 0:
            continue
        similar_customers = item_users.get(history_item, set())
        history_user_count = len(similar_customers)
        if not history_user_count:
            continue

        for candidate, candidate_users in item_users.items():
            if candidate in seen_items:
                continue
            shared_customers = len(similar_customers & candidate_users)
            if shared_customers == 0:
                continue
            similarity = shared_customers / math.sqrt(history_user_count * len(candidate_users))
            scores[candidate] = scores.get(candidate, 0.0) + weight * similarity
            evidence[candidate] = max(evidence.get(candidate, 0), shared_customers)

    ranked = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
    if not ranked:
        raise ValueError(
            "There are no unpurchased products with overlapping customer interactions to recommend."
        )

    return {
        "user_column": user_column,
        "item_column": item_column,
        "customer_value": customer_key,
        "recommendations": [
            {
                "item": item,
                "score": round(scores[item], 4),
                "similar_customer_count": evidence[item],
            }
            for item in ranked
        ],
        "method": "Weighted item similarity from customer co-interactions",
        "interaction_count": int(len(customer_history)),
    }
