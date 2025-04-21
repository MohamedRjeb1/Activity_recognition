import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Define le as a global variable
le = LabelEncoder()

# Charger le dataset
def prepare_data(output_csv):
    df = pd.read_csv(output_csv)

    # Extraire les features et labels
    X = df.drop('label', axis=1).values
    y = df['label'].values

    # Encoder les labels using the global le
    y_encoded = le.fit_transform(y)

    # Paramètres
    sequence_length = 20  # nombre de frames par séquence

    # Construction des séquences cohérentes
    X_sequences = []
    y_sequences = []

    # On parcourt les données pour créer des séquences de frames du même label
    for i in range(len(X) - sequence_length):
        # Vérifier que toutes les frames appartiennent au même label
        if len(set(y[i:i+sequence_length])) == 1:
            X_sequences.append(X[i:i+sequence_length])
            y_sequences.append(y_encoded[i])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    print("Shape des données LSTM : ", X_sequences.shape)  # (nb_sequences, 30, nb_features)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
