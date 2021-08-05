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
        self._epochs = 25
        self.initialize_model()


    def initialize_model(self):
        self._model.add(layers.Conv2D(32, (3, 3), activation='elu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (3, 3), activation='elu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='elu'))
        self._model.add(layers.Conv2D(64, (3, 3), activation='elu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='elu'))
        self._model.add(layers.Conv2D(64, (3, 3), activation='elu'))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(128, activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        self._model.summary()
        
        optimizer = keras.optimizers.Adam(lr=self._learning_rate)
        self._model.compile(optimizer=optimizer,
                            loss=self.loss,
                            metrics=[tf.keras.metrics.AUC(name="auc")])

    def loss(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(tf.argmax(y_true, 1), y_pred)

    def train(self):
        # 6595 = average over all classes
        print("\033[33m Generating dataset. If you are using create_balanced_dataset, this may take a while... \033[00m") 
        #ds = self._ut.create_dataset(self._ut.get_training_names()).repeat(self._epochs)
        #print(ds.element_spec)
        ds = tf.data.experimental.load("./bal_ds", (tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32, name=None), tf.TensorSpec(shape=(None, 15), dtype=tf.int64, name=None))).repeat(self._epochs)
        validation_ds = self._ut.create_dataset(self._ut.get_valid_names())
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, mode='max',
                save_freq='epoch')
        self._history = self._model.fit(ds, epochs=self._epochs,
                                        steps_per_epoch=775,
                                        validation_data=validation_ds,
                                        validation_steps=self._ut.get_validation_steps(),
                                        callbacks=[checkpoint])
                                            #keras.callbacks.EarlyStopping(monitor="loss", mode="min", patience=3)])

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
