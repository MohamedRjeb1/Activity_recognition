

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from src.preprocessing_reconnaissance_activite_data.data_prepare import le
from src.preprocessing_reconnaissance_activite_data.data_prepare import X_train

sequence_length=20;
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
