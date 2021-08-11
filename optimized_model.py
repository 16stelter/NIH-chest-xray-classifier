import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint
import utility
import pickle

class OptunaModel:
    """
    Implementation of the best model according to the optuna study.
    """
    def __init__(self):
        self._model = models.Sequential()
        self._history = 0
        self.initialize_model()
        self._ut = utility.Utility(".")
        self._epochs = 20
        self._learning_rate

    def initialize_model(self):
        """
        Initializes and compiles the model.
        """
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(256, activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        opt = keras.optimizers.Adam(lr=self._learning_rate)
        self._model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])

    def train(self):
        """
        Trains the model, using class weights and checkpoints. Saves the history and the best performing weights.
        """
        ds = self._ut.create_dataset(self._ut.get_training_names()).repeat(self._epochs)
        validation_ds = self._ut.create_dataset(self._ut.get_valid_names())
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max',
                                     save_freq='epoch', save_weights_only=True)
        class_weights = [0.1237545, 0.63923679, 2.22684278, 0.47803896, 2.06822319, 6.58236776,
                         6.04907407, 7.05317139, 1.31780131, 20.02452107, 5.08899708, 6.51670823,
                         2.39743119, 3.55537415, 68.76842105]

        class_weights = {i : class_weights[i] for i in range(len(class_weights))}
        print(class_weights)
        self._history = self._model.fit(ds, epochs=self._epochs,
                                        steps_per_epoch=self._ut.get_steps_per_epoch(),
                                        validation_data=validation_ds,
                                        validation_steps=self._ut.get_validation_steps(),
                                        callbacks=[checkpoint],
                                        class_weight=class_weights)

        with open('history', 'wb') as f:
            pickle.dump(self._history.history, f)

    def predict(self):
        """
        Predicts on the test dataset.
        :return: images, true labels, predicted labels
        """
        ds = self._ut.create_dataset(self._ut.get_test_names())
        x = np.concatenate([x for x, y in ds], axis=0)
        y_test = np.concatenate([y for x, y in ds], axis=0)
        y_pred = self._model.predict(ds)
        return x, y_test, y_pred


if __name__ == "__main__":
    cnn = OptunaModel()
    cnn.train()
    _, y_test, y_pred = cnn.predict()
    cnn._ut.classification_report(y_test, y_pred)
