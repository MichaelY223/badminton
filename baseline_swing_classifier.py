import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data/training/dataset.csv"

# Static per-frame pose angles, plus scale-normalized and windowed wrist motion.
# Raw wrist_x/wrist_y/wrist_displacement are excluded: they're pixel/frame-position
# dependent and don't transfer across videos shot at different distances/crops.
MODEL_FEATURES = [
    "elbow_angle", "shoulder_angle", "wrist_angle", "knee_angle", "other_knee_angle",
    "trunk_rotation", "torso_lean", "contact_height",
    "wrist_displacement_norm", "wrist_displacement_norm_roll_max", "wrist_displacement_norm_roll_std",
]


def main():
    df = pd.read_csv(DATASET_PATH)
    videos = df["video"].unique()

    for test_video in videos:
        train = df[df["video"] != test_video]
        test = df[df["video"] == test_video]

        X_train, y_train = train[MODEL_FEATURES], train["is_swing"]
        X_test, y_test = test[MODEL_FEATURES], test["is_swing"]

        scaler = StandardScaler().fit(X_train)
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(scaler.transform(X_train), y_train)

        probs = clf.predict_proba(scaler.transform(X_test))[:, 1]
        preds = clf.predict(scaler.transform(X_test))

        print(f"=== Train on {sorted(set(videos) - {test_video})}, test on {test_video} ===")
        print(classification_report(y_test, preds, target_names=["not_swing", "swing"]))
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}\n")


if __name__ == "__main__":
    main()
