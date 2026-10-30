# Titanic Survival Predictor

Enter a passenger's details and the app tells you: **SURVIVED** or **DID NOT SURVIVE** (with the percentage chance).

## How to run (Windows)
1. Install Python 3.10 or newer.
2. Double-click `run.bat`. (Or open a terminal in this folder and run `pip install -r requirements.txt`, then `python app.py`.)
3. Your browser opens at http://127.0.0.1:7860. Fill in the form and press **Submit**.

The first run trains the model automatically if `titanic_model.joblib` is missing (about a minute).
To retrain by hand: `python train_model.py` (prints the full comparison and results).

## Problem
Binary classification: predict whether a Titanic passenger survived (1) or not (0).
Metric: F1 with precision and recall (about 38% survived, so accuracy alone is not enough).

## Data
`sns.load_dataset('titanic')` (891 passengers). Removed `alive` (copy of the target, data leakage) and duplicate columns (`class`, `who`, `adult_male`, `embark_town`, `alone`).

## Approach
1. Stratified 80/20 train/test split first (test set used only once).
2. Features: `family_size`, `is_alone`, `has_cabin`.
3. `ColumnTransformer` inside a `Pipeline` (numeric: median + scaling, categorical: mode + one-hot).
4. Compared Dummy, Logistic Regression, kNN, Decision Tree, Naive Bayes, Random Forest with 5-fold CV.
5. Tuned the top 2 with `GridSearchCV`; final model: Random Forest (`max_depth=8`, `n_estimators=200`).

## Results (test set, evaluated once)
| Metric | Score |
|---|---|
| Accuracy | 0.78 |
| Precision | 0.76 |
| Recall | 0.64 |
| F1 | 0.69 |

Confusion matrix: 96 true negatives, 14 false positives, 25 false negatives, 44 true positives.
Most important features: sex, fare, age.

## Limitations
- Small test set (179 passengers), so scores can move with a few predictions.
- The model misses about a third of real survivors (recall 0.64).
- The data reflects 1912 (women and children first, class differences). The model learns that history, not what is fair.
- Missing age was filled with the median; cabin was reduced to yes/no.
- Learning project: not for real decisions.

## Files
- `app.py`: the demo app
- `train_model.py`: full training and evaluation
- `titanic_model.joblib`: saved model
- `metrics.json`: final test results
- `requirements.txt`, `run.bat`
