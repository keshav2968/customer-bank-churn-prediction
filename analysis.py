"""Rebuild all benchmark tables and the retention-capacity figure."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from download import DATA, verify

REPORT = Path(__file__).resolve().parent / "reports"
NUM = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard",
       "IsActiveMember", "EstimatedSalary"]
CAT = ["Geography", "Gender"]


def split(frame, fraction, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    a, b = next(splitter.split(frame, groups=frame.CustomerId))
    return frame.iloc[a], frame.iloc[b]


def features(rows, engineered):
    x = rows[NUM + CAT].copy()
    if engineered:
        # Allow the age slope to change in midlife and later life.
        for age in [35, 45, 55, 65]:
            x[f"age_over_{age}"] = (x.Age - age).clip(lower=0)
        product = x.pop("NumOfProducts").astype(str)
        funded = x.Balance.gt(0).map({True: "funded", False: "zero"})
        x["product_funding"] = product + "_" + funded
        x["product_market"] = product + "_" + x.Geography
        x["active_age"] = x.IsActiveMember * (x.Age - 40)
    return x


def fit(train, engineered):
    x = features(train, engineered)
    transform = ColumnTransformer([
        ("numbers", StandardScaler(), x.select_dtypes("number").columns.tolist()),
        ("categories", OneHotEncoder(handle_unknown="ignore"), x.select_dtypes(exclude="number").columns.tolist()),
    ])
    model = make_pipeline(transform, LogisticRegression(max_iter=1000))
    return model.fit(x, train.Exited)


def best_threshold(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return thresholds[np.argmax(f1)]


def evaluate(name, y, scores, threshold):
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(model=name, threshold=threshold, precision=precision_score(y, pred),
                recall=recall_score(y, pred), f1=f1_score(y, pred),
                roc_auc=roc_auc_score(y, scores), average_precision=average_precision_score(y, scores),
                tn=tn, fp=fp, fn=fn, tp=tp)


def main():
    verify(DATA)
    df = pd.read_csv(DATA)
    trainval, test = split(df, .20, 42)
    train, valid = split(trainval, .25, 43)
    groups = [set(d.CustomerId) for d in (train, valid, test)]
    assert all(groups[i].isdisjoint(groups[j]) for i, j in [(0, 1), (0, 2), (1, 2)])
    REPORT.mkdir(exist_ok=True)
    splits = [(name, len(d), d.CustomerId.nunique(), d.Exited.mean())
              for name, d in zip(["train", "validation", "test"], [train, valid, test])]
    pd.DataFrame(splits, columns=["split", "rows", "customer_ids", "churn_rate"]).to_csv(REPORT/"splits.csv", index=False)
    eda = []
    for field in ["Geography", "NumOfProducts", "IsActiveMember"]:
        for value, group in train.groupby(field):
            eda.append((field, value, len(group), group.Exited.mean()))
    pd.DataFrame(eda, columns=["field", "value", "rows", "churn_rate"]).to_csv(REPORT/"training_segments.csv", index=False)
    # Restrict trimming to extreme positive training balances. Zero is a valid state.
    positive = train.loc[train.Balance > 0, "Balance"]
    q1, q3 = positive.quantile([.25, .75])
    spread = q3 - q1
    low, high = q1 - 3 * spread, q3 + 3 * spread
    clean = train[train.Balance.eq(0) | train.Balance.between(low, high)]
    pd.DataFrame([dict(lower=low, upper=high, removed=len(train)-len(clean))]).to_csv(REPORT/"outliers.csv", index=False)

    metrics = []
    for engineered, name in [(False, "Plain logistic"), (True, "Feature-engineered logistic")]:
        model = fit(clean, engineered)
        val_scores = model.predict_proba(features(valid, engineered))[:, 1]
        threshold = best_threshold(valid.Exited, val_scores)
        scores = model.predict_proba(features(test, engineered))[:, 1]
        metrics.append(evaluate(name, test.Exited, scores, threshold))
    results = pd.DataFrame(metrics)
    results.to_csv(REPORT/"test_metrics.csv", index=False)
    print(results.to_string(index=False))

    ranked = test.assign(score=scores).sort_values(["score", "id"], ascending=[False, True])
    budgets = []
    for fraction in [.01, .05, .10, .20, .30, .50, 1.0]:
        top = ranked.head(int(len(test) * fraction))
        hits = int(top.Exited.sum())
        budgets.append(dict(budget=fraction, contacts=len(top), observed_churners=hits,
                            precision=hits/len(top), capture=hits/test.Exited.sum(),
                            lift=top.Exited.mean()/test.Exited.mean()))
    pd.DataFrame(budgets).to_csv(REPORT/"capacity.csv", index=False)
    cohort = []
    for field in CAT:
        for value, group in ranked.groupby(field):
            pred = group.score >= threshold
            cohort.append((field, value, len(group), group.Exited.mean(), pred.mean(),
                           precision_score(group.Exited, pred), recall_score(group.Exited, pred)))
    pd.DataFrame(cohort, columns=["field", "value", "rows", "churn_rate", "flag_rate",
                                  "precision", "recall"]).to_csv(REPORT/"cohort_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    share = np.arange(len(test)+1) / len(test)
    capture = np.r_[0, ranked.Exited.cumsum() / ranked.Exited.sum()]
    ax.plot(share*100, capture*100, color="#007869", lw=3, label="Risk-ranked outreach")
    ax.plot([0, 100], [0, 100], "--", color="#78868e", label="Random outreach (expected)")
    point = budgets[2]
    ax.scatter(10, point["capture"]*100, color="#bd522d", zorder=3)
    ax.annotate(f'10% capacity reaches {point["capture"]:.0%} of churners',
                (10, point["capture"]*100), xytext=(8, -17), textcoords="offset points")
    ax.set(xlabel="Share of held-out records contacted (%)", ylabel="Share of observed churners reached (%)",
           title="How much churn fits inside an outreach budget?", xlim=(0, 100), ylim=(0, 102))
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(REPORT/"capture.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
