import utility
import numpy as np


class CustomCNN:
    def __init__(self):
        self._ut = utility.Utility(".")

    def generate_model(self):

        raise NotImplementedError

    def conv_layer(self, bias, img, fltr, stride=1):
        # filter shape: w, h, c, n
        fltr_n, fltr_w, fltr_h, fltr_c = fltr.shape
        # image shape: w, h, c
        img_w, img_h, img_c = img.shape

        output_dim = int((img_w - fltr_w) / stride) + 1
        output = np.zeros((fltr_n, output_dim, output_dim))

        for f in range(fltr_n):
            in_x = out_x = 0
            while in_x + fltr_w <= img_w:
                in_y = out_y = 0
                while in_y + fltr_h <= img_h:
                    output[f, out_x, out_y] = np.sum(fltr[f] * img[in_x:in_x + fltr_w, in_y:in_y + fltr_h, :]) + bias[f]
                    in_y += stride
                    out_y += 1
                in_x += stride
                out_x += 1
        return output

    def conv_layer_backprop(self, bwd_in, fwd_in, fltr, stride=1):
        # filter shape: w, h, c, n
        fltr_n, fltr_w, fltr_h, fltr_c = fltr.shape
        # image shape: w, h, c
        img_w, img_h, img_c = fwd_in.shape

        output = np.zeros(fwd_in.shape)
        out_fltr = np.zeros(fltr.shape)
        out_bias = np.zeros((fltr_n,1))

        for f in range(fltr_n):
            in_x = out_x = 0
            while in_x + fltr_w <= img_w:
                in_y = out_y = 0
                while in_y + fltr_h <= img_h:
                    out_fltr[f] = bwd_in[f, out_x, out_y] * fwd_in[:, in_x:in_x+f, in_y:in_y:f]
                    output[:, in_x:in_x+f, in_y:in_y+f] += bwd_in[f, out_x, out_y] * fltr[f]
                    in_y += stride
                    out_y += 1
                in_x += stride
                out_x += 1
            out_bias[f] = np.sum(bwd_in[f])
        return output, out_fltr, out_bias

    def pooling_layer(self, img, kernel_size, stride=1):
        img_w, img_h, img_c = img.shape
        output_dim = int((img_w - kernel_size) / stride) + 1
        output = np.zeros((output_dim, output_dim, img_c))

        for c in range(img_c):
            in_x = out_x = 0
            while in_x + kernel_size <= img_w:
                in_y = out_y = 0
                while in_y + kernel_size <= img_h:
                    output[out_x, out_y, c] = np.max(img[in_x:in_x + kernel_size, in_y:in_y + kernel_size, c])
                    in_y += stride
                    out_y += 1
                in_x += stride
                out_x += 1
        return output

    def pooling_layer_backprop(self, bwd_in, fwd_in, kernel_size, stride):
        img_w, img_h, img_c = fwd_in.shape
        output = np.zeros(fwd_in.shape)

        for c in range(img_c):
            in_x = out_x = 0
            while in_x + kernel_size <= img_w:
                in_y = out_y = 0
                while in_y + kernel_size <= img_h:
                    current = fwd_in[c, in_x:in_x+kernel_size, in_y:in_y+kernel_size]
                    (x, y) = np.unravel_index(np.nanargmax(current), current.shape)
                    output[c, in_x+x, in_y+y] = bwd_in[c, out_x, out_y]

                    in_y += stride
                    out_y += 1
                in_x += stride
                out_x += 1
        return output

    def dense_layer(self, img, weights, bias):
        # 1: Flattening step
        img_w, img_h, img_c = img.shape
        flat = img.reshape((img_w * img_h * img_c, 1))
        # 2: dense/relu layer
        z = weights[0].dot(flat) + bias[0]
        z[z <= 0] = 0   # ReLU
        output = weights[1].dot(z) + bias[1]
        # 3: softmax
        return np.exp(output)/np.sum(np.exp(output))

    def categorical_crossentropy(self, y_hat, y):
        return -np.sum(y * np.log(y_hat))

    def layer_weight_init(self, size):
        #TODO: make smart init
        return np.random.uniform(size=size)



cnn = CustomCNN()
print(np.asarray(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[0]).shape)
print(cnn.pooling_layer([0, 0, 0, 0, 0],
                     np.asarray(next(iter(cnn._ut.create_dataset(cnn._ut.get_training_names())))[0][0]),
                     np.zeros((5, 3, 3, 3))).shape)
