import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
KEYPOINT_CSV_PATH = f"output/keypoints.csv" 
SEQUENCE_LENGTH=30
def train():
    df = pd.read_csv(KEYPOINT_CSV_PATH )
    X = df.drop("label", axis=1).values
    y = df["label"].values
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq, y_seq = [], []
    for i in range(len(X) - SEQUENCE_LENGTH):
        if len(set(y[i:i + SEQUENCE_LENGTH])) == 1:
            X_seq.append(X[i:i + SEQUENCE_LENGTH])
            y_seq.append(y_encoded[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2)

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        LSTM(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    model.save("saved_model/lstm_model.h5")
