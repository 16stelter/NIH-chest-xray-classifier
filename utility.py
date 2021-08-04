import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

class Utility:
    """
    This class provides several utility functions for handling tfrecord files and methods that are common with all
    machine learning tasks.
    """
    def __init__(self, path):
        self._batch_size = 64
        self._filenames = tf.io.gfile.glob(path + "/data/*.tfrec")
        print("\033[92m There are %d total .tfrecord files. \033[00m" % (len(self._filenames)))
        split_ind_l = int(0.7 * len(self._filenames))
        split_ind_r = int(0.9 * len(self._filenames))
        self._training_names, self._valid_names, self._test_names = self._filenames[:split_ind_l], \
                                                                    self._filenames[split_ind_l:split_ind_r], \
                                                                    self._filenames[split_ind_r:-1]
        print("\033[92m I found %d training, %d validation and %d test .tfrecord files. \033[00m" % (
        len(self._training_names), len(self._valid_names), len(self._test_names)))
        self._df = pd.read_csv(path + "/preprocessed_data.csv")
        self._df.rename(columns={'Unnamed: 0': 'image'}, inplace=True)

        print(np.asarray(tf.data.TFRecordDataset(self._test_names)))

        self._train_len = int(np.ceil(sum(1 for _ in tf.data.TFRecordDataset(self._training_names)) / self._batch_size))
        self._valid_len = int(np.ceil(sum(1 for _ in tf.data.TFRecordDataset(self._valid_names)) /self._batch_size))
        self._test_len = int(np.ceil(sum(1 for _ in tf.data.TFRecordDataset(self._test_names)) / self._batch_size))
        print("\033[92m Steps per epoch: " + str(self._train_len) + "\033[00m")
        print("\033[92m Validation steps: " + str(self._valid_len) + "\033[00m")
        print("\033[92m Test steps: " + str(self._test_len) + "\033[00m")

    def read_tfrecord(self, sample):
        """
        Reads a single tfrecord file and turns it into images and labels.

        :param sample: the file to be read
        :return: image: the image array, label: vector with label encodings
        """
        tfrecord_format = {
            "image": tf.io.FixedLenFeature([], tf.string),
        }
        for e in list(self._df.columns)[1:]:
            tfrecord_format[e] = tf.io.FixedLenFeature([], tf.int64)

        sample = tf.io.parse_single_example(sample, tfrecord_format)
        image = self.decode_image(sample["image"])

        label = []
        for i in self._df.columns[1:]:
            label.append(sample[i])

        return image, label

    def decode_image(self, image):
        """
        Turns encoded tfrecord binary into usable jpeg format.

        :param image: encoded image data of a single image
        :return: Decoded image
        """
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, (100, 100))
        image = tf.cast(image, tf.float32)
        return image / 255.0  # normalizing

    def create_dataset(self, path):
        """
        Turns tfrecord files into a tf dataset using the methods above.

        :param path: Paths to the tfrecord files
        :return: The dataset
        """
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(self.read_tfrecord)
        ds = ds.with_options(
            ignore_order
        )
        ds = ds.shuffle(1024)
        ds = ds.batch(self._batch_size)
        return ds
        # ex = next(iter(ds))
        # print(ex[1:])
        # plt.imshow(ex[0] / 255.0)
        # plt.show()

    def create_balanced_dataset(self, path, n_per_class, save=True):
        ds = self.create_dataset(path)
        out_ds = []
        selected_per_class = np.zeros(15)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        for i in range(int(np.ceil(sum(1 for _ in tf.data.TFRecordDataset(path)) / self._batch_size))):
            batch = next(iter(ds))
            for j in range(len(batch[0])):
                if all(x < n_per_class for x in selected_per_class[np.where(batch[1][j] == 1)]):
                    out_ds.append((batch[0][j], batch[1][j]))
                    selected_per_class += np.asarray(batch[1][j])
                    if all(y == n_per_class for y in selected_per_class):
                        output = tf.data.Dataset.from_generator(lambda: ((x, y) for (x, y) in out_ds),
                                                                output_types=(tf.float32, tf.int64),
                                                                output_shapes=((100, 100, 3), (15)))
                        output = output.with_options(ignore_order)
                        output = output.shuffle(1024)
                        output = output.batch(self._batch_size)
                        self._train_len = len(out_ds) / self._batch_size
                        if save:
                            print(output.element_spec)
                            tf.data.experimental.save(output, "./bal_ds")
                        return output
        output = tf.data.Dataset.from_generator(lambda: ((x, y) for (x, y) in out_ds),
                                                output_types=(tf.float32, tf.int64),
                                                output_shapes=((100, 100, 3), (15)))
        output = output.with_options(ignore_order)
        output = output.shuffle(1024)
        output = output.batch(self._batch_size)
        print(selected_per_class)
        self._train_len = len(out_ds) / self._batch_size
        writer = tf.data.experimental.TFRecordWriter('bal_ds.tfrecord')
        writer.write(output)
        return output




    def get_training_names(self):
        return self._training_names

    def get_test_names(self):
        return self._test_names

    def get_valid_names(self):
        return self._valid_names

    def get_steps_per_epoch(self):
        return self._train_len

    def get_validation_steps(self):
        return self._valid_len

    def get_test_steps(self):
        return self._test_len

    def get_batch_size(self):
        return self._batch_size

    def get_column_names(self):
        return list(self._df.columns)[1:]

    def classification_report(self, y_test, y_pred):
        y_pred_classes = np.argmax(y_pred, axis=1)
        print(y_pred_classes)
        print(confusion_matrix(y_test, y_pred_classes))
        print(classification_report(y_test, y_pred_classes))


if __name__ == "__main__":
    ut = Utility(".")
    print(ut.create_balanced_dataset(ut._training_names, 6959))
