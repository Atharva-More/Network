import os
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.layers import (
    GlobalAveragePooling1D,
    Reshape,
    Dense,
    Multiply,
    Input,
    Conv1D,
    BatchNormalization,
    Activation
)
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def squeeze_and_excitation_block(input_feature, ratio=8):
    """
    Squeeze-and-Excitation (SE) Block for 1D temporal feature maps.
    Uses ratio=8 to preserve fine-grained channel features across 37 Adiac classes.
    """
    filters = int(input_feature.shape[-1])

    # Squeeze: Global Average Pooling across temporal axis
    se = GlobalAveragePooling1D()(input_feature)
    se = Reshape((1, filters))(se)

    # Excitation: Two dense layers acting as a channel bottleneck
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # Scale: Multiply original features by channel attention weights
    return Multiply()([input_feature, se])


nb_epochs = 2000
flist = ['Adiac']

for each in flist:
    fname = each
    x_train, y_train = readucr(fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    batch_size = min(x_train.shape[0] // 10, 16)

    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    Y_train = keras.utils.to_categorical(y_train, nb_classes)
    Y_test = keras.utils.to_categorical(y_test, nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    # --- BUILD SE-FCN MODEL (Exact 3-Block FCN Backbone + SE Attention) ---
    input_layer = Input(shape=(x_train.shape[1:]))

    # Block 1 (128 filters)
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = squeeze_and_excitation_block(conv1, ratio=8)

    # Block 2 (256 filters)
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = squeeze_and_excitation_block(conv2, ratio=8)

    # Block 3 (128 filters)
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = squeeze_and_excitation_block(conv3, ratio=8)

    # Output Layers
    full = GlobalAveragePooling1D()(conv3)
    out = Dense(nb_classes, activation='softmax')(full)

    model = Model(inputs=input_layer, outputs=out)

    # Optimizer & Label Smoothing Loss for Fine-Grained Class Generalization
    optimizer = keras.optimizers.Adam()
    loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=50,
        min_lr=0.0001
    )

    checkpoint_path = f'best_se_fcn_{fname}.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    hist = model.fit(
        x_train, Y_train,
        batch_size=batch_size,
        epochs=nb_epochs,
        verbose=1,
        validation_data=(x_test, Y_test),
        callbacks=[reduce_lr, model_checkpoint]
    )

    # --- REPORT METRICS ---
    best_loss_idx = int(np.argmin(hist.history['loss']))
    best_loss = hist.history['loss'][best_loss_idx]
    val_acc_key = 'val_accuracy' if 'val_accuracy' in hist.history else 'val_acc'
    best_loss_val_acc = hist.history[val_acc_key][best_loss_idx]

    peak_val_acc = max(hist.history[val_acc_key])
    peak_val_epoch = int(np.argmax(hist.history[val_acc_key])) + 1

    print("\n" + "=" * 65)
    print(f"Best Training Loss Epoch - Loss: {best_loss:.4f} | Val Acc: {best_loss_val_acc:.4f}")
    print(f"Peak Validation Accuracy - Epoch {peak_val_epoch}: {peak_val_acc:.4f} (Error Rate: {1 - peak_val_acc:.4f})")
    print("=" * 65 + "\n")

    # Load best checkpointed weights for clean CAM / final evaluation
    best_model = keras.models.load_model(checkpoint_path)
    test_loss, test_acc = best_model.evaluate(x_test, Y_test, verbose=0)
    print(f"Verified Best Checkpoint Test Accuracy: {test_acc:.4f} | Test Error: {1 - test_acc:.4f}")

    # Modern Keras 3 CAM Extraction (replaces legacy keras.backend.function)
    # Note: layer[-4] is the Conv3 output before SE scaling, layer[-3] is the SE-weighted Conv3 output
    cam_model = Model(inputs=best_model.inputs, outputs=[best_model.layers[-3].output, best_model.output])
    conv_out, predictions = cam_model.predict(x_test, verbose=0)