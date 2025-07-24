

import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib



DATA_PATH = Path("Enrich3.csv")         
df = pd.read_csv(DATA_PATH)


df = df[df["Disease_Code"] != "D187"].reset_index(drop=True)
df = df[df["Disease_Code"] != "D097"].reset_index(drop=True)
df = df[df["Disease_Code"] != "D067"].reset_index(drop=True)
df = df[df["Disease_Code"] != "D287"].reset_index(drop=True)




MAX_SAMPLES_PER_CLASS = 51

balanced_df = (
    df.groupby("Disease_Code", group_keys=False)
      .apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS),
                                random_state=42))
      .reset_index(drop=True)
)


balanced_df = balanced_df[
    balanced_df["Disease_Code"].map(balanced_df["Disease_Code"].value_counts()) > 1
].reset_index(drop=True)



balanced_df["parsed_symptoms"] = balanced_df["extracted_symptoms"].apply(
    lambda s: [tok.strip() for tok in str(s).split(";") if tok.strip()]
)


mlb = MultiLabelBinarizer()
X_symptoms = mlb.fit_transform(balanced_df["parsed_symptoms"])

symptom_df = pd.DataFrame(X_symptoms, columns=mlb.classes_, dtype="uint8")
final_df   = pd.concat([symptom_df, balanced_df["Disease_Code"]], axis=1)


final_df.to_csv("Clean_numerical2.csv", index=False)


X = final_df.drop("Disease_Code", axis=1)
y = final_df["Disease_Code"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f" Total samples: {len(balanced_df)}")
print("Samples per class (head):\n", balanced_df["Disease_Code"].value_counts().head())


clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,               
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

print("Report:\n",
      classification_report(y_test, y_pred, zero_division=0))

print("Macro‑F1 :", round(f1_score(y_test, y_pred, average="macro"), 3))
print("Weighted‑F1 :", round(f1_score(y_test, y_pred, average="weighted"), 3))


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
joblib.dump(clf, f"Model.pkl")
joblib.dump(mlb, f"Codage.pkl")

print("Model and encoder saved.")
