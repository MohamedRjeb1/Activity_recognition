
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
from keras.src.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Bidirectional,TimeDistributed,LSTM,GlobalAveragePooling2D, Dense)
import datetime as dt
from src.preprocessing.normalize_videos import train_features, train_labels, test_features, test_labels, CLASSES_LIST, \
    SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH
import tensorflow as tf


BATCH_SIZE = 16  # Batch plus grand


def create_lightning_model():
    # Base CNN pré-entraînée gelée
    base_cnn = MobileNetV2(weights='imagenet',
                           include_top=False,
                           input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    base_cnn.trainable = False

    # Modèle séquentiel
    model = Sequential([
        TimeDistributed(base_cnn, input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        TimeDistributed(GlobalAveragePooling2D()),
        Bidirectional(LSTM(32, dropout=0.4, recurrent_dropout=0.4)),
        BatchNormalization(),
        Dense(len(CLASSES_LIST), activation='softmax')
    ])

    # Configuration d'entraînement
    model.compile(
        optimizer=Adam(0.00015),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Générateur d'augmentation temps réel
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=12,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.08,
    brightness_range=[0.92, 1.08],
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0
)


# Application de Mixup pour amélioration des perfs
def mixup_generator(X, y, batch_size=16, alpha=0.2):
    while True:
        indices = np.random.randint(0, len(X), batch_size)
        X_batch = X[indices]
        y_batch = y[indices]

        lam = np.random.beta(alpha, alpha)
        mixed_X = lam * X_batch + (1 - lam) * X_batch[::-1]
        mixed_y = lam * y_batch + (1 - lam) * y_batch[::-1]

        yield mixed_X, mixed_y


# Callbacks
callbacks = [
    EarlyStopping(patience=18, monitor='val_accuracy', mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.4, patience=6, min_lr=1e-6, verbose=1),
    ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
]

# Initialisation
model = create_lightning_model()
model.summary()

# Entraînement
history = model.fit(
    mixup_generator(train_features, train_labels),
    steps_per_epoch=len(train_features) // BATCH_SIZE,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)
# Evaluate the trained model.
model_evaluation_history = model.evaluate(test_features, test_labels)
# Get the loss and accuracy from model_evaluation_history.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

# Save the Model.
model.save(model_file_name)