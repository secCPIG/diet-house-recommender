"""Healthy Diet Recommender: suggests meals from your age, weight, budget, and goal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from recommender.model import KNNRecommender
from recommender.rules import apply_rules, load_catalog

DATA_DIR = Path(__file__).parent.parent / "data"
GOALS = ["lose_weight", "build_muscle", "stay_healthy"]
FEATURES = ["age", "weight_kg", "budget", "vegetarian"] + [f"goal_{g}" for g in GOALS]


def encode(person: dict) -> dict:
    """Turn the goal word into three 0 or 1 columns, because models only understand numbers."""
    if person["goal"] not in GOALS:
        raise ValueError(f"Unknown goal '{person['goal']}'. Choose from: {', '.join(GOALS)}")
    encoded = {key: value for key, value in person.items() if key != "goal"}
    for goal in GOALS:
        encoded[f"goal_{goal}"] = int(person["goal"] == goal)
    return encoded


def build_model() -> KNNRecommender:
    """Load the training data, encode it, and train the model."""
    data = pd.read_csv(DATA_DIR / "diet_training.csv")
    encoded = pd.DataFrame([encode(row) for row in data.to_dict("records")])
    model = KNNRecommender(FEATURES, target_column="meal")
    model.train(encoded)
    return model


def recommend_meals(model: KNNRecommender, catalog: pd.DataFrame, person: dict,
                    top_n: int = 3) -> list[tuple[str, float]]:
    """Model ranks the meals, then the rules remove anything unsafe or too expensive."""
    ranked = model.rank(encode(person))
    avoid = {"meat", "fish"} if person["vegetarian"] else set()
    return apply_rules(ranked, catalog, top_n, max_price=person["budget"], avoid_tags=avoid)


def ask_number(question: str, low: float, high: float) -> float:
    """Keep asking until the user types a number between low and high."""
    while True:
        text = input(question).strip()
        try:
            value = float(text)
        except ValueError:
            print(f"'{text}' is not a number.")
            continue
        if low <= value <= high:
            return value
        print(f"Please enter a number from {low:g} to {high:g}.")


def main() -> None:
    catalog = load_catalog(DATA_DIR / "meals.csv")
    model = build_model()

    person = {
        "age": ask_number("Age: ", 10, 100),
        "weight_kg": ask_number("Weight in kg: ", 30, 250),
        "budget": ask_number("Budget per meal in $: ", 1, 100),
        "vegetarian": int(input("Vegetarian? (y/n): ").strip().lower().startswith("y")),
    }
    print("Goals:", ", ".join(GOALS))
    person["goal"] = input("Your goal: ").strip()

    try:
        picks = recommend_meals(model, catalog, person)
    except ValueError as error:
        print(f"Error: {error}")
        return

    if not picks:
        print("No meals fit. Try a bigger budget.")
        return

    print("\nTop picks for you:")
    for name, confidence in picks:
        meal = catalog.loc[name]
        print(f"  {name}: {meal['calories']} cal, {meal['protein']} g protein,"
              f" ${meal['price']}  (match {confidence:.0%})")


if __name__ == "__main__":
    main()