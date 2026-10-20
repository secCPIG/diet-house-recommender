"""Tests for the model, the rules, and both recommenders. Run with:  python -m pytest"""

import pandas as pd
import pytest

from recommender import diet, house
from recommender.model import KNNRecommender
from recommender.rules import apply_rules, load_catalog


@pytest.fixture(scope="module")
def diet_model():
    return diet.build_model()  # trained once, shared by the tests below


@pytest.fixture(scope="module")
def meals():
    return load_catalog(diet.DATA_DIR / "meals.csv")


def test_rules_remove_banned_and_expensive_items(meals):
    ranked = [("Beef ramen", 0.5), ("Salmon with rice", 0.3), ("Lentil soup", 0.2)]
    safe = apply_rules(ranked, meals, max_price=14, avoid_tags={"meat"})
    # Beef ramen has meat, Salmon costs $15. Only Lentil soup is left.
    assert safe == [("Lentil soup", 0.2)]


def test_vegetarian_never_gets_meat_or_fish(diet_model, meals):
    for goal in diet.GOALS:
        person = {"age": 25, "weight_kg": 80, "budget": 15, "vegetarian": 1, "goal": goal}
        for name, _ in diet.recommend_meals(diet_model, meals, person):
            assert not meals.loc[name, "tags"] & {"meat", "fish"}


def test_recommendations_stay_within_budget(diet_model, meals):
    person = {"age": 30, "weight_kg": 70, "budget": 8, "vegetarian": 0, "goal": "stay_healthy"}
    picks = diet.recommend_meals(diet_model, meals, person)
    assert picks
    assert all(meals.loc[name, "price"] <= 8 for name, _ in picks)


def test_unknown_goal_gives_clear_error():
    with pytest.raises(ValueError, match="Unknown goal"):
        diet.encode({"age": 20, "weight_kg": 60, "budget": 10, "vegetarian": 0, "goal": "get_rich"})


def test_model_must_be_trained_first():
    model = KNNRecommender(["age"], "meal")
    with pytest.raises(RuntimeError):
        model.rank({"age": 20})


def test_missing_columns_give_clear_error():
    model = KNNRecommender(["age", "budget"], "meal")
    with pytest.raises(ValueError, match="budget"):
        model.train(pd.DataFrame({"age": [20, 30], "meal": ["a", "b"]}))


def test_model_beats_random_guessing():
    data = pd.read_csv(diet.DATA_DIR / "diet_training.csv")
    encoded = pd.DataFrame([diet.encode(row) for row in data.to_dict("records")])
    accuracy = KNNRecommender(diet.FEATURES, "meal").evaluate(encoded, k=3)
    # Random guessing would get about 3 out of 12 meals = 25%.
    assert accuracy > 0.5


def test_same_model_works_for_houses():
    model = house.build_model()
    homes = load_catalog(house.DATA_DIR / "houses.csv")
    person = {"age": 22, "budget": 2000, "people": 1, "max_commute": 30}
    picks = house.recommend_homes(model, homes, person)
    assert picks
    assert all(homes.loc[name, "price"] <= 2000 for name, _ in picks)