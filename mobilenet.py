import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint

import utility
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models
from keras.applications.mobilenet import MobileNet


class MNetCNN:
    """
    Mobilenet implementation for the NIH chest xray classification problem.
    """
    def __init__(self):
        self._model = models.Sequential()
        self._ut = utility.Utility(".")
        self._history = 0
        self._learning_rate = 0.00001
        self._epochs = 10
        self.initialize_model()

    def initialize_model(self):
        """
        Generates and compiles the model.
        """
        self._model.add(MobileNet(input_shape=(100, 100, 3),
                                               include_top=False,
                                               weights=None))
        self._model.add(layers.GlobalAveragePooling2D())
        self._model.add(layers.Dense(512))
        self._model.add(layers.Dense(15, activation = 'sigmoid'))
        self._model.compile(optimizer='adam',
                            loss='binary_crossentropy',
                            metrics=['binary_accuracy', 'mae'])

    def train(self):
        """
        Trains the model. Saves weights and history.
        """
        print("\033[33m Generating dataset. If you are using create_balanced_dataset, this may take a while... \033[00m")
        ds = self._ut.create_dataset(self._ut.get_training_names()).repeat(self._epochs)

        print(tf.data.experimental.cardinality(ds))

        validation_ds = self._ut.create_dataset(self._ut.get_valid_names())
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                                     save_freq='epoch')

        class_weights = [0.1237545, 0.63923679, 2.22684278, 0.47803896, 2.06822319, 6.58236776,
                         6.04907407, 7.05317139, 1.31780131, 20.02452107, 5.08899708, 6.51670823,
                         2.39743119, 3.55537415, 68.76842105]
        # These weights have been calculated once, recalculating takes too much time and is not necessary, since the set
        # does not change.

        class_weights = {i : class_weights[i] for i in range(len(class_weights))}
        print(class_weights)
        self._history = self._model.fit(ds, epochs=self._epochs,
                                        steps_per_epoch=self._ut.get_steps_per_epoch(),
                                        validation_data=validation_ds,
                                        validation_steps=self._ut.get_validation_steps(),
                                        callbacks=[checkpoint],
                                        class_weight=class_weights)
        # keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=3)])

        with open('history', 'wb') as f:
            pickle.dump(self._history.history, f)

    def predict(self):
        """
        Makes predictions on the test dataset.
        :return: True labels, predicted labels
        """
        ds = self._ut.create_dataset(self._ut.get_test_names())
        y_test = np.argmax(np.concatenate([y for x, y in ds], axis=0), axis=1)
        y_pred = self._model.predict(ds)
        return y_test, y_pred

if __name__ == "__main__":
    cnn = MNetCNN()
    cnn.train()
    y_test, y_pred = cnn.predict()
    cnn._ut.classification_report(y_test, y_pred)
