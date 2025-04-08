from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (ConvLSTM2D, MaxPooling3D, Dropout,
                                     Flatten, Dense, BatchNormalization)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.preprocessing.normalize_videos import SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES_LIST, train_features, \
    train_labels, test_features, test_labels


def create_enhanced_convlstm_model():
    model = Sequential(name="Enhanced_ConvLSTM")

    # Hyperparamètres optimisés
    reg = l2(1e-4)  # Régularisation L2
    dropout_rate = 0.3  # Augmentation du dropout

    # Block 1
    model.add(ConvLSTM2D(filters=16,
                         kernel_size=(3, 3),
                         activation='swish',  # Meilleure que tanh
                         padding='same',
                         kernel_regularizer=reg,
                         return_sequences=True,
                         input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(dropout_rate))

    # Block 2
    model.add(ConvLSTM2D(filters=32,
                         kernel_size=(3, 3),
                         activation='swish',
                         padding='same',
                         kernel_regularizer=reg,
                         return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(dropout_rate + 0.1))

    # Block 3
    model.add(ConvLSTM2D(filters=64,
                         kernel_size=(3, 3),
                         activation='swish',
                         padding='same',
                         kernel_regularizer=reg,
                         return_sequences=False))  # Dernière séquence
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2)))
    model.add(Dropout(dropout_rate + 0.2))

    # Classification Head améliorée
    model.add(Flatten())
    model.add(Dense(128, activation='swish', kernel_regularizer=reg))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(len(CLASSES_LIST), activation='softmax'))

    # Optimizer configuré
    optimizer = Adam(
        learning_rate=0.0001,
        weight_decay=1e-5,
        clipnorm=1.0
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    model.summary()
    return model


# Callbacks améliorés
callbacks = [
    EarlyStopping(monitor='val_accuracy',
                  patience=15,
                  mode='max',
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=0.5,
                      patience=5,
                      min_lr=1e-6,
                      verbose=1),
    ModelCheckpoint('best_convlstm.keras',
                    save_best_only=True,
                    monitor='val_accuracy')
]

# Initialisation et entraînement
convlstm_model = create_enhanced_convlstm_model()

history = convlstm_model.fit(
    x=train_features,
    y=train_labels,
    epochs=100,  # Plus d'epochs avec early stopping
    batch_size=8,  # Batch size augmenté
    validation_split=0.2,
    shuffle=True,
    callbacks=callbacks
)

# Évaluation finale
test_loss, test_acc = convlstm_model.evaluate(test_features, test_labels)
print(f"\nTest Accuracy: {test_acc:.2%} | Test Loss: {test_loss:.2f}")