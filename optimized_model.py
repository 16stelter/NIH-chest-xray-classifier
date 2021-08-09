import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint
import utility
import pickle

class OptunaModel:
    def __init__(self):
        self._model = models.Sequential()
        self._history = 0
        self.initialize_model()
        self._ut = utility.Utility(".")
        self._epochs = 20

    def initialize_model(self):
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=(100, 100, 3)))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(512, activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        opt = keras.optimizers.Adam(lr=0.00414899434357157)
        self._model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="auc")])

    def train(self):
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
        ds = self._ut.create_dataset(self._ut.get_test_names())
        y_test = np.argmax(np.concatenate([y for x, y in ds], axis=0), axis=1)
        y_pred = self._model.predict(ds)
        return y_test, y_pred

if __name__ == "__main__":
    cnn = OptunaModel()
    cnn.train()
    y_test, y_pred = cnn.predict()
    cnn._ut.classification_report(y_test, y_pred)
