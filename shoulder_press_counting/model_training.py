#this code is executed  for  single time to create the two models (model_shoulder_press and model2_shoulder_press)
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib


# Charger les données (exemple)
df = pd.read_csv('shoulder_press_counting/annotated_angles.csv')

# Extraire les caractéristiques et les labels
X = df[['right_shoulder','left_shoulder']].values
y = df['label'].values

# Encodage des labels (si 'label' est sous forme de texte, on le transforme en numérique)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Normalisation des données (si nécessaire)
scaler = StandardScaler()

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=1000)  # max_iter peut être ajusté si nécessaire
model.fit(X_train, y_train)

# Créer et entraîner le modèle RandomForest
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)

# Prédiction et évaluation
y_pred = model2.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# Save the model to a file
joblib.dump(model, 'shoulder_press_counting/model_shoulder_press.pkl')
joblib.dump(label_encoder,'shoulder_press_counting/label_encoder_shoulder_press.pkl')
joblib.dump(model2, 'shoulder_press_counting/model2_shoulder_press.pkl')



