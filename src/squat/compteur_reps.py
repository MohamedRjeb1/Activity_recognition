# counter_model.py

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1) Charger les jeux de données
train_df = pd.read_csv("./dataset/train.csv")
test_df  = pd.read_csv("./dataset/test.csv")

# 2) Ne conserver que les lignes "down" ou "up"
train_df = train_df[ train_df["label"].isin(["down","up"]) ]
test_df  = test_df[  test_df["label"].isin(["down","up"]) ]

# 3) Préparer X et y (0=down, 1=up)
X_train = train_df.drop(columns=["label"])
y_train = train_df["label"].map({"down":0, "up":1})

X_test  = test_df.drop(columns=["label"])
y_test  = test_df["label"].map({"down":0, "up":1})

# 4) Entraîner un RandomForest binaire
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5) Évaluer
y_pred = clf.predict(X_test)
print("=== Rapport de classification binaire ===")
print(classification_report(
    y_test, y_pred,
    target_names=["down","up"],
    zero_division=0
))

# 6) Sauvegarder le modèle
with open("stage_model.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Modèle binaire down/up sauvegardé dans stage_model.pkl")
