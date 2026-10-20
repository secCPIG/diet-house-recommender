"""Hard rules the model is never allowed to break.

The ML model ranks options, but it can make mistakes. These checks make sure
a vegetarian never gets meat and nobody gets something over their budget.
"""

from __future__ import annotations

import pandas as pd


def load_catalog(csv_path) -> pd.DataFrame:
    """Read meals.csv or houses.csv and turn the "a;b;c" tags into a set."""
    catalog = pd.read_csv(csv_path)
    catalog["tags"] = catalog["tags"].fillna("").apply(
        lambda text: {t.strip() for t in text.split(";") if t.strip()}
    )
    return catalog.set_index("name")


def is_allowed(item: pd.Series, max_price: float | None = None,
               avoid_tags: set[str] | None = None) -> bool:
    """True if one catalog item passes every hard rule."""
    if max_price is not None and item["price"] > max_price:
        return False
    if avoid_tags and item["tags"] & avoid_tags:
        return False
    return True


def apply_rules(ranked: list[tuple[str, float]], catalog: pd.DataFrame, top_n: int = 3,
                max_price: float | None = None,
                avoid_tags: set[str] | None = None) -> list[tuple[str, float]]:
    """Keep the model's order, drop anything that breaks a rule, return the top_n."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    safe = [
        (name, confidence)
        for name, confidence in ranked
        if name in catalog.index and is_allowed(catalog.loc[name], max_price, avoid_tags)
    ]
    return safe[:top_n]