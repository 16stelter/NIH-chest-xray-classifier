import keras.callbacks
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

import utility
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models


class TfCNN:
    def __init__(self):
        self._model = models.Sequential()
        self._ut = utility.Utility(".")
        self.initialize_model()
        self._history = 0

    def initialize_model(self):
        self._model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(64, activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        self._model.summary()

        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=[tf.keras.metrics.AUC(name="auc")])

        # self._history = self._model.fit(train_images, train_labels, epochs=10,
        #                    validation_data=(test_images, test_labels))

    def train(self):
        ds = self._ut.create_dataset(self._ut.get_training_names()).repeat(self._ut.get_batch_size())
        validation_ds = self._ut.create_dataset(self._ut.get_valid_names())
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',
                save_freq=1000)
        self._history = self._model.fit(ds, epochs=10,
                                        steps_per_epoch=self._ut.get_steps_per_epoch(),
                                        validation_data=validation_ds,
                                        validation_steps=self._ut.get_validation_steps(),
                                        callbacks=[checkpoint,
                                            keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=3)])

        with open('history', 'wb') as f:
            pickle.dump(self._history.history, f)

    def predict(self):
        ds = self._ut.create_dataset(self._ut.get_test_names())
        y_test = np.argmax(np.concatenate([y for x, y in ds], axis=0), axis=1)
        y_pred = self._model.predict(ds)
        return y_test, y_pred

if __name__ == "__main__":
    cnn = TfCNN()
    cnn.train()
    y_test, y_pred = cnn.predict()
    cnn._ut.classification_report(y_test, y_pred)
