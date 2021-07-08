import utility
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class TfCNN:
    def __init__(self):
        self._model = models.Sequential()
        self._ut = utility.Utility(".")
        self.initialize_model()
        self._history = 0

    def initialize_model(self):
        self._model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(768, 768, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        self._model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(64, activation='relu'))
        self._model.add(layers.Dense(15))

        self._model.summary()

        self._model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        #self._history = self._model.fit(train_images, train_labels, epochs=10,
        #                    validation_data=(test_images, test_labels))


    def train(self):
        ds = self._ut.create_dataset(self._ut.get_training_names())
        validation_ds = self._ut.create_dataset(self._ut.get_test_names())

        self._history = self._model.fit(ds, epochs=10,
                            validation_data=validation_ds)

    def predict(self):
        pass

if __name__ == "__main__":
    cnn = TfCNN()
    cnn.train()