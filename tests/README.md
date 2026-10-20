# Healthy Diet and House Recommendation System

A machine learning recommender built with scikit-learn. It suggests meals based on a
person's age, weight, budget, health goal, and diet, and the same model is reused to
suggest homes based on age, budget, household size, and commute.

## How it works

1. **The model ranks.** A K Nearest Neighbors classifier finds the people in the training
   data most similar to you and ranks every option by how well it fit them.
2. **The rules protect.** Hard rules then remove anything that must never be shown, like
   meat for a vegetarian or anything over your budget. The model can make mistakes,
   but the rules make sure those mistakes are never harmful.

```
user input → encode → scale → KNN model → ranked options → safety rules → top 3
```

## Results

Tested on 20% of the data the model never saw during training:

| Recommender | Top 1 accuracy | Top 3 accuracy |
|-------------|----------------|----------------|
| Meals (12 options) | 44% | 87% |
| Homes (8 options)  | 67% | 95% |

Random guessing would give about 25% top 3 accuracy for meals.

## About the data

I don't have real user data, so `data/make_data.py` generates 800 example people for each
recommender using common sense rules plus randomness. The model only sees the examples,
not the rules. The meal nutrition values and home listings are rough sample values,
not official data.

## Project structure

```
recommender/
    model.py    # KNNRecommender: scaling + KNN in a scikit-learn Pipeline
    rules.py    # hard rules (budget, diet) applied after the model
    diet.py     # meal recommender menu
    house.py    # home recommender menu
data/           # catalogs, training data, and the script that makes it
tests/          # pytest tests
```

## How to run

```
python -m pip install -r requirements.txt
python -m recommender.diet
python -m recommender.house
python -m pytest
```

## What I would improve next

* Train on real user feedback instead of synthetic data
* Tune the number of neighbors with cross validation
* Add more meals and real nutrition data