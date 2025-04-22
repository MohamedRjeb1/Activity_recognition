from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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


# Architecture du modèle LSTM
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
    model.add(LSTM(128, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(le.classes_), activation='softmax'))
    # Compilation
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model




def train_model(model, X_train, y_train, X_test, y_test):
# Entraînement avec batch_size
  model.fit(X_train, y_train,
          epochs=20,
          batch_size=20,  # par exemple
          validation_data=(X_test, y_test))
if __name__=="__main__":
  output_csv = 'keypoints7.csv'
  X_train, y_train, X_test, y_test=prepare_data(output_csv)
  #create_lstm_model
  model=create_lstm_model(X_train.shape)
#train the model
  train_model(model, X_train, y_train, X_test, y_test)
#save the model
  model.save('final_lstm_mode.keras')