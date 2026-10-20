"""Make example training data, because I don't have real user data.

Each row is one made up person (age, weight, budget, ...) and the meal or home
that best fits them. The "best fit" comes from simple common sense rules plus
some randomness, so the data looks a bit like real, messy choices.
The model never sees these rules. It only sees the examples and has to learn the pattern.

Run once:  python data/make_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
rng = np.random.default_rng(seed=42)  # fixed seed, so the data is the same every time


def has_tag(tags: str, tag: str) -> bool:
    return tag in tags.split(";")


def best_meal(meals: pd.DataFrame, age, weight_kg, budget, goal, vegetarian) -> str:
    options = meals[meals["price"] <= budget]
    if vegetarian:
        options = options[~options["tags"].str.contains("meat|fish")]
    if options.empty:
        options = meals.nsmallest(1, "price")

    if goal == "lose_weight":
        score = -options["calories"] + 3 * options["protein"]
    elif goal == "build_muscle":
        score = 10 * options["protein"] - abs(options["calories"] - weight_kg * 8)
    else:  # stay_healthy: calories close to a normal meal for this body weight
        score = -abs(options["calories"] - weight_kg * 7)

    if age >= 40:  # older people get a bonus for high fiber food
        score = score + 80 * options["tags"].apply(has_tag, tag="high_fiber")

    score = score + rng.normal(0, 40, len(options))  # people are not perfect
    return options.loc[score.idxmax(), "name"]


def best_home(houses: pd.DataFrame, age, budget, people, max_commute) -> str:
    options = houses[houses["price"] <= budget]
    if options.empty:
        options = houses.nsmallest(1, "price")

    score = (
        -30 * abs(options["bedrooms"] - np.ceil(people / 2))   # right size
        - 2 * np.maximum(options["commute_minutes"] - max_commute, 0)  # long commute
        + options["price"] / 100                               # nicer if you can afford it
    )
    if age < 25:
        score = score + 20 * options["tags"].apply(has_tag, tag="near_transit")
    if people >= 3:
        score = score + 25 * options["tags"].apply(has_tag, tag="yard")

    score = score + rng.normal(0, 10, len(options))
    return options.loc[score.idxmax(), "name"]


def main(rows: int = 800) -> None:
    meals = pd.read_csv(DATA_DIR / "meals.csv")
    houses = pd.read_csv(DATA_DIR / "houses.csv")

    diet_rows = []
    for _ in range(rows):
        person = {
            "age": int(rng.integers(18, 66)),
            "weight_kg": round(float(rng.normal(72, 13)), 1),
            "budget": int(rng.integers(6, 18)),
            "goal": str(rng.choice(["lose_weight", "build_muscle", "stay_healthy"])),
            "vegetarian": int(rng.random() < 0.3),
        }
        person["meal"] = best_meal(meals, **person)
        diet_rows.append(person)

    house_rows = []
    for _ in range(rows):
        person = {
            "age": int(rng.integers(18, 66)),
            "budget": int(rng.integers(1100, 4200)),
            "people": int(rng.integers(1, 6)),
            "max_commute": int(rng.integers(10, 45)),
        }
        person["home"] = best_home(houses, **person)
        house_rows.append(person)

    pd.DataFrame(diet_rows).to_csv(DATA_DIR / "diet_training.csv", index=False)
    pd.DataFrame(house_rows).to_csv(DATA_DIR / "house_training.csv", index=False)
    print(f"Saved {rows} diet rows and {rows} house rows.")


if __name__ == "__main__":
    main()