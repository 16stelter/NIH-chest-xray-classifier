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
        self._history = 0
        self._learning_rate = 0.00001
        self._epochs = 10
        self.initialize_model()


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
        
        optimizer = keras.optimizers.Adam(lr=self._learning_rate)
        self._model.compile(optimizer=optimizer,
                            loss='categorical_crossentropy',
                            metrics=[tf.keras.metrics.AUC(name="auc")])

    def train(self):
        # 6595 = average over all classes
        print("\033[33m Generating dataset. If you are using create_balanced_dataset, this may take a while... \033[00m") 
        ds = self._ut.create_balanced_dataset(self._ut.get_training_names(), 6595, save=True).repeat(self._epochs)
        validation_ds = self._ut.create_dataset(self._ut.get_valid_names())
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                save_freq='epoch')
        self._history = self._model.fit(ds, epochs=self._epochs,
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
