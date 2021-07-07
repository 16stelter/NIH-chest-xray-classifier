import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


class Utility:
    def __init__(self, path):
        self._filenames = tf.io.gfile.glob(path + "/data/*.tfrec")
        split_ind = int(0.9 * len(self._filenames))
        self._training_names, self._test_names = self._filenames[:split_ind], self._filenames[split_ind:]
        print("I found %d training and %d test .tfrecord files." % (len(self._training_names), len(self._test_names)))
        self._df = pd.read_csv(path + "/preprocessed_data.csv")

    def read_tfrecord(self, sample):
        tfrecord_format = {
                "image": tf.io.FixedLenFeature([], tf.string),
            }
        for e in list(self._df.columns)[1:]:
            tfrecord_format[e] = tf.io.FixedLenFeature([], tf.int64)

        sample = tf.io.parse_single_example(sample, tfrecord_format)
        image = self.decode_image(sample["image"])

        label = []
        for i in range(len(self._df.columns[1:])):
            label.append(sample[self._df.columns[i+1]])

        return image, label

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

        ex = next(iter(ds))
        print(ex[1:])
        plt.imshow(ex[0] / 255.0)
        plt.show()



ut = Utility(".")
ut.create_dataset(ut._training_names[0])