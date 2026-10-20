"""House Recommender: the same KNN model and rules, used for homes instead of meals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender.diet import ask_number
from recommender.model import KNNRecommender
from recommender.rules import apply_rules, load_catalog

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES = ["age", "budget", "people", "max_commute"]


def build_model() -> KNNRecommender:
    data = pd.read_csv(DATA_DIR / "house_training.csv")
    model = KNNRecommender(FEATURES, target_column="home")
    model.train(data)
    return model


def recommend_homes(model: KNNRecommender, catalog: pd.DataFrame, person: dict,
                    top_n: int = 3) -> list[tuple[str, float]]:
    ranked = model.rank(person)
    return apply_rules(ranked, catalog, top_n, max_price=person["budget"])


def main() -> None:
    catalog = load_catalog(DATA_DIR / "houses.csv")
    model = build_model()

    person = {
        "age": ask_number("Age: ", 16, 100),
        "budget": ask_number("Max monthly rent in $: ", 500, 20000),
        "people": ask_number("How many people will live there: ", 1, 12),
        "max_commute": ask_number("Longest commute you're okay with (minutes): ", 5, 120),
    }

    picks = recommend_homes(model, catalog, person)
    if not picks:
        print("No homes fit. Try a bigger budget.")
        return

    print("\nTop picks for you:")
    for name, confidence in picks:
        home = catalog.loc[name]
        print(f"  {name}: ${home['price']}/month, {home['bedrooms']} bed,"
              f" {home['commute_minutes']} min commute  (match {confidence:.0%})")


if __name__ == "__main__":
    main()