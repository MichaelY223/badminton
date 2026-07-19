"""Leave-one-video-out swing/no-swing baseline on window features.

Gradient-boosted trees replace the earlier per-frame logistic regression: window
statistics capture temporal shape, and trees capture feature interactions
(e.g. "high wrist speed AND extended elbow AND trunk rotation together") that a
linear model structurally cannot.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATASET_PATH = "data/training/windows.csv"

META_COLS = {"video", "player", "center_frame", "is_swing", "stroke_type"}


def main():
    df = pd.read_csv(DATASET_PATH)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    videos = df["video"].unique()

    if len(videos) < 2:
        print(f"Only {len(videos)} video(s) in the dataset - leave-one-video-out needs at least 2. "
              "Reporting apparent (train=test) performance instead; treat it as an upper bound only.")
        train_test_pairs = [(df, df, videos[0] if len(videos) else "n/a")]
    else:
        train_test_pairs = [(df[df["video"] != v], df[df["video"] == v], v) for v in videos]

    for train, test, held_out in train_test_pairs:
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        clf.fit(train[feature_cols], train["is_swing"])

        probs = clf.predict_proba(test[feature_cols])[:, 1]
        preds = clf.predict(test[feature_cols])

        print(f"=== Held-out video: {held_out} "
              f"({int(test['is_swing'].sum())} swings / {len(test)} windows) ===")
        print(classification_report(test["is_swing"], preds, target_names=["not_swing", "swing"],
                                    zero_division=0))
        if test["is_swing"].nunique() > 1:
            print(f"ROC-AUC: {roc_auc_score(test['is_swing'], probs):.3f}\n")


if __name__ == "__main__":
    main()
