import pickle
from operator import add

from tqdm import tqdm

import utility
import numpy as np
from scipy.special import softmax


class CustomCNN:
    def __init__(self):
        self._ut = utility.Utility(".")
        self._kernel_size = 2
        self._stride = 1
        self._cost_history = []

    def model(self, img, label, bias, filters):

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
        loss = self.categorical_crossentropy(prediction, label)

        # backpropagation
        dout = prediction - np.asarray(label).reshape((15, 1))
        dflat, dw8, db8, dw7, db7 = self.dense_layer_backprop(dout, flat, filters[6:8], bias[6:8], z)

        dconv6 = dflat.reshape(conv6.shape)
        dconv6[conv6 <= 0] = 0
        dconv5, df6, db6 = self.conv_layer_backprop(dconv6, conv5, filters[5])
        dconv5[conv5 <= 0] = 0
        dpool2, df5, db5 = self.conv_layer_backprop(dconv5, pool2, filters[4])
        dconv4 = self.pooling_layer_backprop(dpool2, conv4)
        dconv4[conv4 <= 0] = 0
        dconv3, df4, db4 = self.conv_layer_backprop(dconv4, conv3, filters[3])
        dconv3[conv3 <= 0] = 0
        dpool1, df3, db3 = self.conv_layer_backprop(dconv3, pool1, filters[2])
        dconv2 = self.pooling_layer_backprop(dpool1, conv2)
        dconv2[conv2 <= 0] = 0
        dconv1, df2, db2 = self.conv_layer_backprop(dconv2, conv1, filters[1])
        dconv1[conv1 <= 0] = 0
        dimg, df1, db1 = self.conv_layer_backprop(dconv1, img[0], filters[0])

        weight_gradients = [df1, df2, df3, df4, df5, df6, dw7, dw8]
        bias_gradients = [db1, db2, db3, db4, db5, db6, db7, db8]

        return weight_gradients, bias_gradients, loss

    def conv_layer(self, bias, img, fltr):
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
                    # here we need normalization, else output gets too big for softmax
                    normalized = val / (fltr_w * fltr_h)
                    output[out_x, out_y, f] = normalized
                    in_y += self._stride
                    out_y += 1
                in_x += self._stride
                out_x += 1
        return output

    def conv_layer_backprop(self, bwd_in, fwd_in, fltr):
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
            out_bias[f] = np.sum(bwd_in[f])
        return output, out_fltr, out_bias

    def pooling_layer(self, img):
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
        dweight2 = img.dot(z.T)
        dbias2 = np.sum(img, axis=1).reshape(bias[1].shape)
        dz = weights[1].T.dot(img)
        dz[z <= 0] = 0
        dweight1 = dz.dot(flat.T)
        dbias1 = np.sum(dz, axis=1).reshape(bias[0].shape)
        dflat = weights[0].T.dot(dz)
        return dflat, dweight2, dbias2, dweight1, dbias1

    def categorical_crossentropy(self, y_hat, y):
        y_hat[y_hat == 0] = 10 ** -10
        return -np.sum(y * np.log(y_hat))

    def layer_weight_init(self, size):
        # TODO: make smart init
        return np.random.uniform(size=size)

    def gradient_descent(self, alpha, batch, weight_gradients, bias_gradients):
        dwg = [0] * 8
        dbg = [0] * 8
        cost = 0

        for i in range(batch[0].shape[0]-1):
            img = np.expand_dims(np.asarray(batch[0][i]), axis=0)
            wg, bg, loss = self.model(img, batch[1][i], bias_gradients, weight_gradients)

            dwg = list(map(add, dwg, wg))
            dbg = list(map(add, dbg, bg))

            cost += loss
        for j in range(len(dwg)):
            weight_gradients[j] = weight_gradients[j] - alpha * dwg[j]
            bias_gradients[j] = bias_gradients[j] - alpha * dbg[j]

        cost = cost/len(batch)
        self._cost_history.append(cost)

        return weight_gradients, bias_gradients

    def train(self, alpha, epochs, path, n_filters):
        weights = []  # TODO: refactor this
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
                batch = np.asarray(next(iter(self._ut.create_dataset(self._ut.get_training_names()))))
                weights, bias = self.gradient_descent(alpha, batch, weights, bias)
                t.set_description("Cost: %f" % self._cost_history[-1])

            with open(path, 'wb') as f:
                pickle.dump((weights, bias), f)


cnn = CustomCNN()
# print("first layer out" + str(cnn.conv_layer([0, 0, 0, 0, 0], np.asarray(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[0][0]), np.zeros((5, 3, 3, 3))).shape))
# out = cnn.conv_layer([0, 0, 0, 0, 0], np.asarray(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[0][0]), np.zeros((5, 3, 3, 3)))
# print("second layer out" + str(cnn.conv_layer([0, 0, 0, 0, 0], out, np.zeros((5, 3, 3, 3)))))
cnn.train(0.01, 10, "./weights", 2)
# print(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[1])
# print(cnn.pooling_layer([0, 0, 0, 0, 0],
#                     np.asarray(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[0][0]),
#                     np.zeros((5, 3, 3, 3))).shape)
