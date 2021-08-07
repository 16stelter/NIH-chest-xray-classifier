import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint


class OptunaModel:
    def __init__(self):
        self._model = models.Sequential
        self._history = 0
        self.initialize_model()

    def initialize_model(self):
        self._model.add(layers.Conv2D(32,
                                      (2, 2),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32,
                                      (2, 2),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32,
                                      (2, 2),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32,
                                      (2, 2),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32,
                                      (2, 2),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(256, activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        opt = keras.optimizers.Adam(lr=0.003460515573512333)
        self._model.compile(optimizer=opt, loss=self.loss, metrics=[tf.keras.metrics.AUC(name="auc")])

    def train(self):
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max',
                                     save_freq='epoch')
        self._history = self._model.fit(self.ds, epochs=2,
                                        steps_per_epoch=tf.data.experimental.cardinality(self.ds).numpy(),
                                        callbacks=[checkpoint])

