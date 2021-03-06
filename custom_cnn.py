import pickle
from operator import add

from tqdm import tqdm

import utility
import numpy as np
from scipy.special import softmax
from multiprocessing import Pool, cpu_count, Process, Queue, Manager


class CustomCNN:
    """
    Implements a CNN from scratch, providing all basic functionality necessary for that.
    """
    def __init__(self):
        self._ut = utility.Utility(".")
        self._kernel_size = 2
        self._stride = 1
        self._cost_history = []
        self._manager = Manager()
        self._q = self._manager.Queue()
        self._dwgq = self._manager.Queue()
        self._dbgq = self._manager.Queue()

        self._dwg = [0] * 8
        self._dbg = [0] * 8
        self._cost = 0

    def model(self, img, label, bias, filters):
        """
        Defines the CNN model by calling the prediction function and then running this prediction through
        backpropagation.

        :param img: A single image processed by the network
        :param label: The corresponding true label
        :param bias: Bias of the network
        :param filters: Weights of the network
        :return: New weights, new bias, loss
        """
        prediction, z, flat, layers = self.predict(bias, img, filters)

        loss = self.categorical_crossentropy(prediction, label)

        # backpropagation
        dout = prediction - np.asarray(label).reshape((15, 1))
        dflat, dw8, db8, dw7, db7 = self.dense_layer_backprop(dout, flat, filters[6:8], bias[6:8], z)

        dconv6 = dflat.reshape(layers[-1].shape)
        dconv6[layers[-1] <= 0] = 0
        dconv5, df6, db6 = self.conv_layer_backprop(dconv6, layers[-2], filters[5])
        dconv5[layers[-2] <= 0] = 0
        dpool2, df5, db5 = self.conv_layer_backprop(dconv5, layers[-3], filters[4])
        dconv4 = self.pooling_layer_backprop(dpool2, layers[-4])
        dconv4[layers[-4] <= 0] = 0
        dconv3, df4, db4 = self.conv_layer_backprop(dconv4, layers[-5], filters[3])
        dconv3[layers[-5] <= 0] = 0
        dpool1, df3, db3 = self.conv_layer_backprop(dconv3, layers[-6], filters[2])
        dconv2 = self.pooling_layer_backprop(dpool1, layers[-7])
        dconv2[layers[-7] <= 0] = 0
        dconv1, df2, db2 = self.conv_layer_backprop(dconv2, layers[-8], filters[1])
        dconv1[layers[-8] <= 0] = 0
        dimg, df1, db1 = self.conv_layer_backprop(dconv1, img[0], filters[0])

        weight_gradients = [df1, df2, df3, df4, df5, df6, dw7, dw8]
        bias_gradients = [db1, db2, db3, db4, db5, db6, db7, db8]

        return weight_gradients, bias_gradients, loss

    def predict(self, bias, img, filters):
        """
        Defines the forward pass of the network.

        :param bias: The bias of the network
        :param img: Image for which the label should be predicted
        :param filters: Weights of the network
        :return: Prediction, states of the layers
        """
        # forward pass
        conv1 = self.conv_layer(bias[0], img[0], filters[0])
        conv1[conv1 <= 0] = 0
        conv2 = self.conv_layer(bias[1], conv1, filters[1])
        conv2[conv2 <= 0] = 0
        pool1 = self.pooling_layer(conv2)
        conv3 = self.conv_layer(bias[2], pool1, filters[2])
        conv3[conv3 <= 0] = 0
        conv4 = self.conv_layer(bias[3], conv3, filters[3])
        conv4[conv4 <= 0] = 0
        pool2 = self.pooling_layer(conv4)
        conv5 = self.conv_layer(bias[4], pool2, filters[4])
        conv5[conv5 <= 0] = 0
        conv6 = self.conv_layer(bias[5], conv5, filters[5])
        conv6[conv6 <= 0] = 0

        # calculate loss
        prediction, z, flat = self.dense_layer(conv6, filters[6:8], bias[6:8])  # i.e. 6 and 7
        layers = [conv1, conv2, pool1, conv3, conv4, pool2, conv5, conv6]
        return prediction, z, flat, layers

    def conv_layer(self, bias, img, fltr):
        """
        Convolves over the image.

        :param bias: Bias of the layer
        :param img: The image to be processed
        :param fltr: The weights of the layer
        :return: Prediction
        """
        # filter shape: w, h, c, n
        fltr_n, fltr_w, fltr_h, fltr_c = fltr.shape
        # image shape: w, h, c
        img_w, img_h, img_c = img.shape

        output_dim = int((img_w - fltr_w) / self._stride) + 1
        output = np.zeros((output_dim, output_dim, fltr_n))

        for f in range(fltr_n):
            in_x = out_x = 0
            while in_x + fltr_w <= img_w:
                in_y = out_y = 0
                while in_y + fltr_h <= img_h:
                    val = np.sum(fltr[f] * img[in_x:in_x + fltr_w, in_y:in_y + fltr_h, :]) + bias[f]
                    normalized = val / (fltr_w * fltr_h)
                    output[out_x, out_y, f] = normalized
                    in_y += self._stride
                    out_y += 1
                in_x += self._stride
                out_x += 1
        return output

    def conv_layer_backprop(self, bwd_in, fwd_in, fltr):
        """
        Backpropagation of the convolutional layer.

        :param bwd_in: State of the previous layer in the backpropagation
        :param fwd_in: State of the corresponding layer in the forward pass
        :param fltr: Weights of the layer
        :return: New state, new weights, new bias
        """
        # filter shape: w, h, c, n
        fltr_n, fltr_w, fltr_h, fltr_c = fltr.shape
        # image shape: w, h, c
        img_w, img_h, img_c = fwd_in.shape

        output = np.zeros(fwd_in.shape)
        out_fltr = np.zeros(fltr.shape)
        out_bias = np.zeros((fltr_n, 1))

        for f in range(fltr_n):
            in_x = out_x = 0
            while in_x + fltr_w <= img_w:
                in_y = out_y = 0
                while in_y + fltr_h <= img_h:
                    out_fltr[f] = bwd_in[out_x, out_y, f] * fwd_in[in_x:in_x+fltr_w, in_y:in_y+fltr_w, :]
                    output[in_x:in_x+fltr_h, in_y:in_y+fltr_h, :] += bwd_in[out_x, out_y, f] * fltr[f]
                    in_y += self._stride
                    out_y += 1
                in_x += self._stride
                out_x += 1

            out_bias[f] = np.sum(bwd_in[:, :, f])
        return output, out_fltr, out_bias

    def pooling_layer(self, img):
        """
        Performs a max-pooling operation on the image.

        :param img: The image to be reduced
        :return: Reduced image
        """
        img_w, img_h, img_c = img.shape
        output_dim = int((img_w - self._kernel_size) / self._kernel_size) + 1
        output = np.zeros((output_dim, output_dim, img_c))

        for c in range(img_c):
            in_x = out_x = 0
            while in_x + self._kernel_size <= img_w:
                in_y = out_y = 0
                while in_y + self._kernel_size <= img_h:
                    output[out_x, out_y, c] = np.max(img[in_x:in_x + self._kernel_size, in_y:in_y + self._kernel_size, c])
                    in_y += self._kernel_size
                    out_y += 1
                in_x += self._kernel_size
                out_x += 1
        return output

    def pooling_layer_backprop(self, bwd_in, fwd_in):
        """
        Backpropagation through the pooling layer.

        :param bwd_in: State of the previous layer in the backpropagation
        :param fwd_in: State of the corresponding layer in the forward pass
        :return: New state
        """
        img_w, img_h, img_c = fwd_in.shape
        output = np.zeros(fwd_in.shape)

        for c in range(img_c):
            in_x = out_x = 0
            while in_x + self._kernel_size <= img_w:
                in_y = out_y = 0
                while in_y + self._kernel_size <= img_h:
                    current = fwd_in[in_x:in_x+self._kernel_size, in_y:in_y+self._kernel_size, c]
                    (x, y) = np.unravel_index(np.nanargmax(current), current.shape)
                    output[in_x+x, in_y+y, c] = bwd_in[out_x, out_y, c]
                    in_y += self._kernel_size
                    out_y += 1
                in_x += self._kernel_size
                out_x += 1
        return output

    def dense_layer(self, img, weights, bias):
        """
        The final dense/fully connected layer. Flattens the input, calculates a dense layer with relu activation, and
        calculates a softmax.

        :param img: Image to be processed
        :param weights: Weights of the layer
        :param bias: Bias of the layer
        :return: Prediction, State of the first dense layer, State of the flattening layer
        """
        # 1: Flattening step
        img_w, img_h, img_c = img.shape
        flat = img.reshape((img_w * img_h * img_c, 1))
        # 2: dense/relu layer
        z = (weights[0].dot(flat) + bias[0]) / (weights[0][0].shape[0] + 1)
        z[z <= 0] = 0   # ReLU
        output = (weights[1].dot(z) + bias[1]) / (weights[1][0].shape[0] + 1)
        # 3: softmax
        return softmax(output), z, flat

    def dense_layer_backprop(self, img, flat, weights, bias, z):
        """
        Backpropagates through the dense layer.

        :param img: The image at the end of the prediction step
        :param flat: The state of the flattening layer
        :param weights: Weights of the layer
        :param bias: Bias of the layer
        :param z: State at the first dense layer
        :return: Result of the backpropagation, weights and biases of each layer
        """
        dweight2 = img.dot(z.T)
        dbias2 = np.sum(img, axis=1).reshape(bias[1].shape)
        dz = weights[1].T.dot(img)
        dz[z <= 0] = 0
        dweight1 = dz.dot(flat.T)
        dbias1 = np.sum(dz, axis=1).reshape(bias[0].shape)
        dflat = weights[0].T.dot(dz)
        return dflat, dweight2, dbias2, dweight1, dbias1

    def categorical_crossentropy(self, y_hat, y):
        """
        Calculates a categorical crossentropy.
        :param y_hat: Prediction
        :param y: True label
        :return: Crossentropy
        """
        y_hat[y_hat == 0] = 10 ** -10
        return -np.sum(y * np.log(y_hat))

    def layer_weight_init(self, size):
        """
        Initializes the weights of the layers
        :param size: Shape the weight matrix should have.
        :return: Weight matrix of shape size.
        """
        # TODO: make smarter init
        return np.random.uniform(size=size)

    def gradient_descent(self, alpha, batch, weight_gradients, bias_gradients):
        """
        Calculates a gradient descent. Uses multiprocessing to speed up the process, generating as many threads as
        there are samples in the batch.

        :param alpha: Learning rate
        :param batch: The batch that is supposed to be processed
        :param weight_gradients: Current weight gradients
        :param bias_gradients: Current bias gradients
        :return: New weight gradients, new bias gradients
        """
        self._dwg = [0] * 8
        self._dbg = [0] * 8
        self._cost = 0

        workers = []
        for i in range(batch[0].shape[0]-1):
            p = Process(target=self.mp_gd, args=(batch, weight_gradients, bias_gradients, i))
            workers.append(p)
            p.start()


        for p in workers:
            self._cost += self._q.get()

            self._dwg = list(map(add, self._dwg, self._dwgq.get()))
            self._dbg = list(map(add, self._dbg, self._dbgq.get()))

            p.join()

        for j in range(len(self._dwg)):
            weight_gradients[j] = weight_gradients[j] - alpha * self._dwg[j]
            bias_gradients[j] = bias_gradients[j] - alpha * self._dbg[j]
        cost = self._cost/len(batch)
        self._cost_history.append(cost)

        return weight_gradients, bias_gradients

    def mp_gd(self, batch, weight_gradients, bias_gradients, i):
        """
        Helper method for Gradient Descent. Places return values in a queue instead of returning them directly.

        :param batch: The current batch
        :param weight_gradients: The current weight gradients
        :param bias_gradients: The current bias gradients
        :param i: Specifies which sample of the batch should be processed by this worker
        """
        img = np.expand_dims(np.asarray(batch[0][i]), axis=0)
        wg, bg, loss = self.model(img, batch[1][i], bias_gradients, weight_gradients)

        self._dwgq.put(wg)
        self._dbgq.put(bg)
        self._q.put(loss)

    def train(self, alpha, epochs, path, n_filters):
        """
        Trains the network.

        :param alpha: Learning rate
        :param epochs: Number of epochs
        :param path: Where results should be stored
        :param n_filters: Number of filters per layer
        """
        weights = []
        bias = []
        weights.append(self.layer_weight_init((n_filters, 3, 3, 3)))
        bias.append(self.layer_weight_init((n_filters, 1)))
        for i in range(5):
            weights.append(self.layer_weight_init((n_filters, 3, 3, n_filters)))
            bias.append(self.layer_weight_init((n_filters, 1)))
        weights.append(self.layer_weight_init((128, 324*n_filters)))
        bias.append(self.layer_weight_init((128, 1)))
        weights.append(self.layer_weight_init((15, 128)))
        bias.append(self.layer_weight_init((15, 1)))

        for e in range(epochs):
            t = tqdm(range(self._ut.get_steps_per_epoch()))
            for j in t:
                batch = next(iter(self._ut.create_dataset(self._ut.get_training_names())))
                weights, bias = self.gradient_descent(alpha, batch, weights, bias)
                t.set_description("Cost: %f" % self._cost_history[-1])

            with open(path + "/weights", 'wb') as f:
                pickle.dump((weights, bias), f)
            with open(path + "/history", 'wb') as f:
                pickle.dump(self._cost_history, f)

    def predict_mp(self, bias, img, weights):
        """
        Helper method to predict labels with multiprocessing.

        :param bias: Bias of the network
        :param img: Image to be predicted
        :param weights: Weights of the network
        :return:
        """
        p, _, _, _ = self.predict(bias, img, weights)

        self._q.put((p, img[1]))
 
if __name__ == "__main__":
    cnn = CustomCNN()
    # cnn.train(0.01, 4, ".", 8)
    ds = cnn._ut.create_dataset(cnn._ut.get_test_names())
    with open ("./experiments/2/weights", 'rb') as f:
        (weights, bias) = pickle.load(f)
    t = tqdm(range(cnn._ut.get_test_steps()))
    prediction = []
    labels = []
    for j in t:
        batch = next(iter(ds))
        workers = []
        for i in range(batch[0].shape[0]-1):
            p = Process(target=cnn.predict_mp, args=(bias, (batch[0][i], batch[1][i]), weights))
            workers.append(p)
            p.start()
        for p in workers:
            (y_pred, y_test) = cnn._q.get()
            prediction.append(y_pred)
            labels.append(np.argmax(y_test))
            p.join()
    cnn._ut.classification_report(labels, prediction)
