import matplotlib.pyplot as plt
import numpy as np
import utility

class Visualizer:
    def __init__(self):
        self._ut = utility.Utility(".")
        return

    def sample_and_show(self, path):
        ds = self._ut.create_dataset(path)
        (x, y) = next(iter(ds))
        self.show_sample_grid(x[0:25], y[0:25])

    def show_sample_grid(self, x, y, y_pred=None):
        plt.figure(figsize=(25, 25))
        grid = (np.floor(np.sqrt(len(x))))
        for n in range(len(x)):
            plt.subplot(grid, grid, n+1)
            plt.axis("off")
            if y_pred:
                plt.title(self.generate_title(y[n], y_pred[n]))
            else:
                plt.title(self.generate_title(y[n], None))
            plt.imshow(x[n])
        plt.show()

    def generate_title(self, y, y_pred=None):
        column_names = self._ut.get_column_names()
        label = []
        pred = []
        for i in range(len(column_names)):
            if y[i]:
                label.append(column_names[i])
            if y_pred:
                if y_pred[i]:
                    pred.append(column_names[i])
        if y_pred:
            return "Label: " + " + ".join(label) + "\n \n Prediction: " + " + ".join(pred)
        return "Label: " + " + ".join(label)
if __name__ == "__main__":
    viz = Visualizer()
    viz.sample_and_show(viz._ut.get_test_names())