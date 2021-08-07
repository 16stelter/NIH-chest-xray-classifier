import keras
import numpy as np
import optuna
from optuna.samplers import TPESampler
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import utility


class Objective(object):
    def __init__(self, calib_ds):
        self.ds = calib_ds
        self._model = models.Sequential()

    def __call__(self, trial):
        n_additional_layers = trial.suggest_int('n_additional_layers', 0, 8)
        n_filters = trial.suggest_categorical('n_filters', [16, 32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        n_dense = trial.suggest_categorical('n_dense', [64, 128, 256, 512, 1024])
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        learning_rate = trial.suggest_float('learning_rate', 1 * 10 ** -8, 0.1)

        dict_params = {'n_additional_layers': n_additional_layers,
                       'n_filters': n_filters,
                       'kernel_size': kernel_size,
                       'n_dense': n_dense,
                       'batch_size': batch_size,
                       'learning_rate': learning_rate}
        self._model = models.Sequential()
        self._model.add(layers.Conv2D(dict_params['n_filters'],
                                      (dict_params['kernel_size'], dict_params['kernel_size']),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(dict_params['n_filters'],
                                      (dict_params['kernel_size'], dict_params['kernel_size']),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.MaxPooling2D((2, 2)))
        for i in range(dict_params['n_additional_layers']):
            self._model.add(layers.Conv2D(dict_params['n_filters'],
                                          (dict_params['kernel_size'], dict_params['kernel_size']),
                                          activation='relu',
                                          input_shape=(100, 100, 3)))
            if i+1 % 2 == 0:
                self._model.add(layers.MaxPooling2D((2, 2)))

        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(dict_params['n_dense'], activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        opt = keras.optimizers.Adam(lr=dict_params['learning_rate'])
        self._model.compile(optimizer=opt, loss=self.loss, metrics=[tf.keras.metrics.AUC(name="auc")])

        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='max',
                                     save_freq='epoch')
        print(tf.data.experimental.cardinality(self.ds).numpy())
        print(dict_params)
        self._history = self._model.fit(self.ds, epochs=2,
                                        steps_per_epoch=tf.data.experimental.cardinality(self.ds).numpy()/dict_params['batch_size'],
                                        callbacks=[checkpoint])

        loss = np.min(self._history.history['loss'])
        return loss

    def loss(self, y_true, y_pred):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(tf.argmax(y_true, 1), y_pred)


max_epochs = 1000
early_stop = 10
lr_epochs = 5
opt_direction = 'minimize'
n_random = 25
max_time = 6*60*60

_ut = utility.Utility(".")
#cal_ds = _ut.create_balanced_dataset(_ut.get_training_names(), 110)
cal_ds = tf.data.experimental.load("./cal_ds", (tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32, name=None), tf.TensorSpec(shape=(None, 15), dtype=tf.int64, name=None))).repeat(1000)

objective = Objective(cal_ds)

study = optuna.create_study(direction=opt_direction, sampler=TPESampler(n_startup_trials=n_random))

study.optimize(objective, timeout=max_time)

result = study.trials_dataframe()
result.to_pickle("./optuna/results.pkl")
result.to_csv("./optuna/results.csv")
