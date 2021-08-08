import keras
import numpy as np
import optuna
from optuna.samplers import TPESampler
from tensorflow.keras import layers, models
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import utility


class Objective(object):
    """
    Optuna study objective.
    """
    def __init__(self, calib_ds):
        self.ds = calib_ds
        self._model = models.Sequential()

    def __call__(self, trial):
        # Initialize optimizable parameters with suitable ranges
        n_additional_layers = trial.suggest_int('n_additional_layers', 0, 8)
        n_filters = trial.suggest_categorical('n_filters', [16, 32, 64, 128])
        kernel_size = trial.suggest_int('kernel_size', 2, 4)
        n_dense = trial.suggest_categorical('n_dense', [64, 128, 256, 512, 1024])
        learning_rate = trial.suggest_float('learning_rate', 1 * 10 ** -8, 0.1)

        dict_params = {'n_additional_layers': n_additional_layers,
                       'n_filters': n_filters,
                       'kernel_size': kernel_size,
                       'n_dense': n_dense,
                       'learning_rate': learning_rate}

        # Generate model according to parameters of current round
        self._model = models.Sequential()
        self._model.add(layers.Conv2D(dict_params['n_filters'],
                                      (dict_params['kernel_size'], dict_params['kernel_size']),
                                      activation='relu',
                                      input_shape=(100, 100, 3)))
        self._model.add(layers.Conv2D(dict_params['n_filters'],
                                      (dict_params['kernel_size'], dict_params['kernel_size']),
                                      activation='relu'))
        self._model.add(layers.MaxPooling2D((2, 2)))
        for i in range(dict_params['n_additional_layers']):
            self._model.add(layers.Conv2D(dict_params['n_filters'],
                                          (dict_params['kernel_size'], dict_params['kernel_size']),
                                          activation='relu'))
            if i+1 % 2 == 0:
                self._model.add(layers.MaxPooling2D((2, 2)))

        self._model.add(layers.Flatten())
        self._model.add(layers.Dense(dict_params['n_dense'], activation='relu'))
        self._model.add(layers.Dense(15, activation='sigmoid'))

        opt = keras.optimizers.Adam(lr=dict_params['learning_rate'])

        # For some reason there sometimes is a visual bug in the loss function during epoch 1.
        # This does not affect the learning, however. Later epochs display the loss correctly.
        self._model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name="auc")])

        # Initialize training parameters
        filepath = "./weights"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min',
                                     save_freq='epoch', save_weights_only=True)
        print(tf.data.experimental.cardinality(self.ds).numpy())
        print(dict_params)

        self._history = self._model.fit(self.ds, epochs=2,
                                        steps_per_epoch=tf.data.experimental.cardinality(self.ds).numpy()/64,
                                        callbacks=[checkpoint])

        loss = np.min(self._history.history['loss'])
        return loss


max_epochs = 1000  # Very high number, because we want to stop after a time.
early_stop = 10  # early stop if 10 runs each produced worse results.
lr_epochs = 5  # epochs to alter learning rate
opt_direction = 'minimize'
n_random = 25  # Random trials before the optimization begins
max_time = 5*60*60  # 5 hours

_ut = utility.Utility(".")
cal_ds = tf.data.experimental.load("./cal_ds", (tf.TensorSpec(shape=(None, 100, 100, 3), dtype=tf.float32, name=None), tf.TensorSpec(shape=(None, 15), dtype=tf.int64, name=None))).repeat(1000)

objective = Objective(cal_ds)

study = optuna.create_study(direction=opt_direction, sampler=TPESampler(n_startup_trials=n_random))

study.optimize(objective, timeout=max_time)

result = study.trials_dataframe()
result.to_pickle("./optuna/results.pkl")
result.to_csv("./optuna/results.csv")
