import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class Utility:
    def __init__(self, path):
        self._filenames = tf.io.gfile.glob(path + "/data/*.tfrec")
        split_ind = int(0.9 * len(self._filenames))
        self._training_names, self._test_names = self._filenames[:split_ind], self._filenames[split_ind:]
        print("I found %d training and %d test .tfrecord files." % (len(self._training_names), len(self._test_names)))

    def read_tfrecord(self, sample):
        tfrecord_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
            }

        sample = tf.io.parse_single_example(sample, tfrecord_format)
        sample = self.decode_image(sample["image"])
        return sample

    def decode_image(self, image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        return image

    def create_dataset(self, path):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        ds = tf.data.TFRecordDataset(path)
        ds = ds.map(self.read_tfrecord)
        ds = ds.with_options(
            ignore_order
        )
        plt.imshow(next(iter(ds)) / 255.0)
        plt.show()

    def load_csv(self, file):
        df = pd.read_csv(file)
        return df


ut = Utility(".")
print(ut.load_csv("./preprocessed_data.csv")["Unnamed: 0"][0])
ut.create_dataset(ut._training_names[0])