"""
Titanic Survival Prediction: full training script.

Steps: load data -> remove leaky/duplicate columns -> stratified split ->
add features -> Pipeline (preprocessing + model) -> compare models with 5-fold CV ->
tune top 2 -> evaluate ONCE on the test set -> save the final model.

Run:  python train_model.py
"""
import os
import json
import joblib
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "titanic_model.joblib")
METRICS_PATH = os.path.join(HERE, "metrics.json")


def add_features(data):
    data = data.copy()
    data["family_size"] = data["sibsp"] + data["parch"] + 1
    data["is_alone"] = (data["family_size"] == 1).astype(int)
    data["has_cabin"] = data["deck"].notnull().astype(int)
    return data.drop(columns=["deck"])


def train(verbose=True):
    df = sns.load_dataset("titanic")
    # 'alive' is a copy of the target (leakage); the rest duplicate other columns
    df = df.drop(columns=["alive", "class", "who", "adult_male", "embark_town", "alone"])

    X = df.drop(columns=["survived"])
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_test = add_features(X_train), add_features(X_test)

    numeric = ["age", "fare", "sibsp", "parch", "family_size", "is_alone", "has_cabin"]
    categorical = ["pclass", "sex", "embarked"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])

    models = {
        "Dummy (baseline)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "kNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = {}
    for name, m in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", m)])
        s = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1"])
        rows[name] = {"accuracy": s["test_accuracy"].mean(), "f1": s["test_f1"].mean()}
    comparison = pd.DataFrame(rows).T.round(3).sort_values("f1", ascending=False)
    if verbose:
        print("\n5-fold CV comparison (training set only):\n", comparison)

    grids = {
        "Logistic Regression": {"model__C": [0.1, 1, 10]},
        "kNN": {"model__n_neighbors": [3, 5, 7, 11], "model__weights": ["uniform", "distance"]},
        "Decision Tree": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 10]},
        "Naive Bayes": {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
        "Random Forest": {"model__n_estimators": [100, 200], "model__max_depth": [5, 8, None],
                          "model__min_samples_leaf": [1, 3]},
    }
    top2 = comparison.drop("Dummy (baseline)").head(2).index.tolist()
    tuned = {}
    for name in top2:
        pipe = Pipeline([("prep", preprocessor), ("model", models[name])])
        gs = GridSearchCV(pipe, grids[name], cv=cv, scoring="f1").fit(X_train, y_train)
        tuned[name] = gs
        if verbose:
            print(f"Tuned {name}: CV F1 = {gs.best_score_:.3f}  {gs.best_params_}")

    best_name = max(tuned, key=lambda n: tuned[n].best_score_)
    final = tuned[best_name].best_estimator_

    # Test set used ONCE, on the final model only
    pred = final.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics = {
        "final_model": best_name,
        "best_params": {k: (v if v is not None else "None") for k, v in tuned[best_name].best_params_.items()},
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }
    if verbose:
        print("\nFINAL TEST RESULTS:", json.dumps(metrics, indent=2))

    joblib.dump(final, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print("\nSaved:", MODEL_PATH)
    return final, metrics


if __name__ == "__main__":
    train()
