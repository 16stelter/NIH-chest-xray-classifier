import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle
import utility
import optimized_model

class Visualizer:
    """
    Various methods to visualize data.
    """
    def __init__(self):
        self._ut = utility.Utility(".")
        return

    def sample_and_show(self, path):
        """
        Draw a sample from the dataset. Display it as a 5x5 grid with labels added.
        :param path: Path from which the dataset should be generated.
        """
        ds = self._ut.create_dataset(path)
        (x, y) = next(iter(ds))
        self.show_sample_grid(x[0:25], y[0:25])

    def predict_and_show(self, weight_path):
        model = optimized_model.OptunaModel()
        model._model.load_weights(weight_path)
        x, y, y_pred =model.predict()
        self.show_sample_grid(x[0:25], y[0:25], y_pred[0:25])

    def show_sample_grid(self, x, y, y_pred=None):
        """
        Shows samples in a grid.

        :param x: The image vector
        :param y: The lable vector
        :param y_pred: Optional predicted labels
        """
        plt.figure(figsize=(25, 25))
        grid = (np.floor(np.sqrt(len(x))))
        for n in range(len(x)):
            plt.subplot(grid, grid, n+1)
            plt.axis("off")
            if y_pred.any():
                plt.title(self.generate_title(y[n], y_pred[n]))
            else:
                plt.title(self.generate_title(y[n], None))
            plt.imshow(x[n])
        plt.tight_layout(h_pad=5)
        plt.subplots_adjust(top=0.95)
        plt.show()

    def generate_title(self, y, y_pred=None):
        """
        Helper method to make the labels human readable.

        :param y: True labels
        :param y_pred: Optional predicted labels
        :return: Human readable combination of the labels.
        """
        column_names = self._ut.get_column_names()
        label = []
        pred = []
        for i in range(len(column_names)):
            if y[i]:
                label.append(column_names[i])
        if y_pred.any():
            pred.append(column_names[np.argmax(y_pred)])
        if y_pred.any():
            return "Label: " + " + ".join(label) + " \n Prediction: " + " + ".join(pred)
        return "Label: " + " + ".join(label)

    def plot_confusion_matrix(self, matrix):
        """
        Plots a confusion matrix.
        :param matrix: A matrix of shape NxN
        """
        df_cm = pd.DataFrame(matrix, index=[i for i in range(15)],
                             columns=[i for i in range(15)])
        plt.figure(figsize=(10,7))
        sn.heatmap(df_cm, annot=False, vmin=0, vmax=1)
        plt.show()


    def plot_history(self, path):
        """
        Plots the loss, auc, val_loss and val_auc history from a pickle file.
        :param path: Path to the pickle
        """
        with open(path, 'rb') as f:
            history = pickle.load(f)
        print(history)
        plt.plot(history['loss'])
        plt.plot(history['auc'])
        plt.plot(history['val_loss'])
        plt.plot(history['val_auc'])
        plt.legend(['loss', 'auc', 'val_loss', 'val_auc'])
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.show()


if __name__ == "__main__":
    viz = Visualizer()
    # viz.plot_confusion_matrix(matrix)
    viz.predict_and_show("./weights")
    # viz.plot_history("./history")
    # viz.sample_and_show(viz._ut.get_test_names())
