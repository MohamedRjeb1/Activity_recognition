from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


from src.preprocessing.data_prepare import X_train, y_train, X_test, y_test, sequence_length, y

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Vérification
print("Classes disponibles:", le.classes_)

# Architecture du modèle LSTM
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(sequence_length, X_train.shape[2])))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(le.classes_), activation='softmax'))


# Compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement avec batch_size
model.fit(X_train, y_train,
          epochs=30,
          batch_size=20,  # par exemple
          validation_data=(X_test, y_test))

model.save('lstm_et_pose_estim2.keras')
