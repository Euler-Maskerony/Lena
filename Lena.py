# Lena - Learning and assEmbly tool for NeurAl networks

import numpy as np
from math import e, ceil
import os
import datetime


def ext_dot(a, b):
    res = np.zeros((a.shape[0], a.shape[1], b.shape[1]), dtype='float64')
    for batch_el in range(res.shape[0]):
        for col in range(res.shape[1]):
            for row in range(res.shape[2]):
                res[batch_el, col, row] = a[batch_el, col] * b[batch_el, row]
    return res


def gen_mul(a, b):
    for batch_el in range(b.shape[0]):
        for row in range(b.shape[1]):
            a[batch_el, row] *= b[batch_el, row]
    return a


class Optimizer:
    def __init__(self, optimizer_type, model):
        self.model = model
        self.optimizer_type = optimizer_type
        self.opt_dic = {
            'gradient_descent': self.GradientDescent
        }
        self.available_kwargs = [['jac_w', 'jac_b', 'step']]

    def __call__(self, **kwargs):
        assert kwargs.keys() not in self.available_kwargs
        return self.opt_dic[self.optimizer_type](**kwargs)

    def GradientDescent(self, jac_w, jac_b, step):
        model_comp = list(filter(lambda layer: not isinstance(layer, (Activation, MaxPooling, AveragePooling, Dropout)), self.model))
        for lay in range(1, len(model_comp) - 1):
            model_comp[lay].w = np.reshape(np.reshape(model_comp[lay].w, -1) - step * jac_w[len(model_comp) - 2 - lay],
                                           model_comp[lay].shape_w)
            if model_comp[lay].use_bias:
                model_comp[lay].b = model_comp[lay].b - step * jac_b[len(model_comp) - 2 - lay]
        return self.model


class Loss:
    def __init__(self):
        self.acc = None
        self.model_results = None
        self.reference_results = None


class BinaryCrossEntropy(Loss):
    def f(self, x, y):
        x = np.transpose(x, (0, 2, 3, 1))
        res = np.zeros_like(x, dtype='float64')
        x += 10**(-10)
        for batch_el in range(np.shape(res)[0]):
            res[batch_el] = -y[batch_el] * np.log(x[batch_el]) - (1 - y[batch_el]) * np.log(1 - x[batch_el])
            res[batch_el] = np.sum(res[batch_el])
        res = np.average(res)
        self.results(x, y)
        return res

    @staticmethod
    def df(x, y):
        x = np.transpose(x, (0, 2, 3, 1))
        res = np.zeros_like(x, dtype='float64')
        x += 10**(-10)
        for batch_el in range(np.shape(res)[0]):
            res[batch_el] = (-y[batch_el] / (x[batch_el] + 10**(-7)) + (1 - y[batch_el]) / (1 - x[batch_el] - 10**(-7))) / np.size(res[batch_el])
        return np.transpose(res, (0, 3, 1, 2))

    def accuracy(self):
        self.acc = np.zeros(np.shape(self.model_results)[0], dtype=bool)
        for batch_el in range(np.shape(self.model_results)[0]):
            self.acc[batch_el] = np.array_equal(self.model_results[batch_el], self.reference_results[batch_el])
        self.acc = np.average(self.acc)
        return self.acc

    def results(self, x, y):
        self.model_results = np.argmax(np.reshape(x, (np.shape(x)[0], -1)), axis=1)
        self.reference_results = np.argmax(np.reshape(y, (np.shape(y)[0], -1)), axis=1)


class Sigmoid:
    @staticmethod
    def f(x):
        return (1 + e ** (-x)) ** (-1)

    @staticmethod
    def df(x):
        return e ** (-x) * (1 + e ** (-x)) ** (-2)


class Tanh:
    @staticmethod
    def f(x):
        return np.tanh(x)

    @staticmethod
    def df(x):
        return np.cosh(x) ** (-2)


class Relu:
    @staticmethod
    def max_0(arr):
        if arr >= 0:
            return arr
        else:
            return 0

    @staticmethod
    def dmax_0(arr):
        if arr >= 0:
            return 1
        else:
            return 0

    def f(self, x):
        return np.vectorize(self.max_0)(x)

    def df(self, x):
        return np.vectorize(self.dmax_0)(x)

class Softmax:
    def f(self, x):
        init_shape = np.shape(x)
        x = np.reshape(x, (init_shape[0], -1))
        self.add = np.max(x, axis=1)
        x = np.transpose(x)
        return np.reshape(np.transpose(e**(x+self.add)/np.sum(e**(x+self.add), axis=0)), init_shape)

    def df(self, x):
        init_shape = np.shape(x)
        x = np.transpose(np.reshape(x, (init_shape[0], -1)))
        c = -e**(x+self.add) + np.sum(e**(x+self.add), axis=0)
        return np.reshape(np.transpose(e**(x+self.add)*c/(c + e**(x+self.add))**2), init_shape)


class Initializer:
    @staticmethod
    def uniform(*args):  # args[0] - shape
        weights = np.random.uniform(size=args[0])
        return weights

    @staticmethod
    def xavier(*args):  # args[0] - shape; args[1] - next_lay
        weights = np.random.uniform(
            -np.sqrt(6) / np.sqrt(args[0][1] + args[1]),
            np.sqrt(6) / np.sqrt(args[0][1] + args[1]),
            size=args[0]
        )
        return weights

    @staticmethod
    def xe(*args):  # args[0] - shape
        weights = np.random.normal(
            0,
            np.sqrt(2 / args[0][1]),
            size=args[0]
        )
        return weights

    @staticmethod
    def none(*args):
        return np.zeros(args[0])


class Layer:
    initializer_dic = {
        'uniform': Initializer.uniform,
        'xavier': Initializer.xavier,
        'xe': Initializer.xe,
        'None': Initializer.none
    }


class Activation(Layer):
    def __init__(self, act_func):
        self.act_func = act_func
        self.x = None
        self.y = None
        self.dy = None
        self.der_w = None
        self.der_b = None
        self.jac_to_go = None
        self.type = 'Activation_' + self.act_func
        self.activation_dic = {
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'relu': Relu(),
            'softmax': Softmax()
        }

    def ForwardProp(self, inputs):
        self.x = inputs
        self.y = self.activation_dic[self.act_func].f(self.x)
        self.dy = self.activation_dic[self.act_func].df(self.x)

    def BackProp(self, der_w, der_b, jac):
        jac = np.reshape(jac, np.shape(self.y))
        self.jac_to_go = jac * self.dy
        self.der_w = der_w
        self.der_b = der_b


class Dense(Layer):
    def __init__(self, out_dim, use_bias=True):
        """Fully connected layer.

        Args:
            out_dim (int): Number of neurons in layer.
            use_bias (bool, optional): Flag for enabling/disabling biases. Defaults to True.
        """
        self.out_dim = out_dim
        self.channels = 1
        self.in_dim = None
        self.use_bias = use_bias
        self.x = None
        self.w = None
        self.shape_w = None
        self.b = None
        self.y = None
        self.dy_dw = None
        self.dy_db = None
        self.der_w = None
        self.der_b = None
        self.jac_to_go = None
        self.type = 'Dense_' + str(self.out_dim) + '_' + str(self.use_bias)

    def ForwardProp(self, inputs):
        self.x = inputs  # (batch_size; input; 1; 1)
        self.y = np.transpose(
                np.transpose(np.dot(self.w, np.transpose(self.x, (0, 2, 1, 3))), (1, 2, 3, 0)) + self.b * int(self.use_bias),
                (0, 3, 1, 2)
                )
        self.dy_dw = np.transpose(np.ones((np.shape(self.w)[0], np.shape(self.x)[0], np.shape(self.w)[1]), dtype='float64') * self.x[:, :, 0, 0], (1, 0, 2))
        self.dy_db = np.ones((np.shape(self.x)[0], np.shape(self.b)[0]), dtype='float64') * int(self.use_bias)

    def BackProp(self, der_w, der_b, jac):
        jac = jac[:, :, 0, 0]
        df_dw = np.reshape(np.average(np.transpose(np.transpose(self.dy_dw, (2, 0, 1)) * jac, (1, 2, 0)), axis=0), -1)
        df_db = np.reshape(np.average(self.dy_db * jac, axis=0), -1)
        der_w.append(df_dw)
        der_b.append(df_db)
        self.der_w = der_w
        self.der_b = der_b
        self.jac_to_go = np.transpose(np.dot(np.transpose(self.w), np.transpose(jac)))
        self.jac_to_go = self.jac_to_go[:, :, np.newaxis, np.newaxis]

    def Initializer(self, prev_lay, next_lay, init_type):
        init = super().initializer_dic[init_type]
        self.in_dim = prev_lay.out_dim
        self.w = np.zeros((self.out_dim, self.in_dim), dtype='float64')
        self.b = np.zeros(self.out_dim, dtype='float64')
        self.shape_w = np.shape(self.w)
        self.w = init(self.shape_w, next_lay.out_dim)


class MaxPooling(Layer):
    def __init__(self, pool_size):
        """Maximum pooling layer.

        Args:
            pool_size (tuple): 2-d tuple. Adjust size of window pooling will hold in.
        """
        self.pool_size = pool_size
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.x = None
        self.der_w = None
        self.der_b = None
        self.type = 'MaxPooling_' + '%dx%d' % (pool_size[0], pool_size[1])

    def ForwardProp(self, inputs):
        self.x = inputs
        self.y = np.zeros((*np.shape(self.x)[:2], np.shape(self.x)[2] // self.pool_size[0], np.shape(self.x)[3] // self.pool_size[1]), dtype='float64')
        self.dy = np.zeros(np.shape(self.x), dtype='float64')
        for batch in range(np.shape(self.y)[0]):
            for chn in range(np.shape(self.y)[1]):
                for row in range(np.shape(self.y)[2]):
                    for col in range(np.shape(self.y)[3]):
                        self.y[batch, chn, row, col] = np.amax(
                            self.x[
                                batch,
                                chn,
                                row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                                col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                                ],
                            axis=(0,1)
                            )
                        mask = np.zeros(self.pool_size)
                        mask[
                            np.unravel_index(
                                np.argmax(
                                    self.x[
                                    batch,
                                    chn,
                                    row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                                    ]
                                ),
                                self.pool_size
                            )] = 1
                        self.dy[
                            batch,
                            chn,
                            row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                            col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                        ] = mask


    def BackProp(self, der_w, der_b, jac):
        self.der_w = der_w
        self.der_b = der_b
        self.jac_to_go = np.zeros_like(self.x, dtype='float64')
        for row in range(np.shape(jac)[2]):
            for col in range(np.shape(jac)[3]):
                self.jac_to_go[
                    :,
                    :,
                    row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                ] = np.transpose(np.transpose(self.dy[
                    :,
                    :,
                    row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]], (2,3,0,1)) * jac[:, :, row, col], (2,3,0,1))


class AveragePooling(Layer):
    def __init__(self, pool_size):
        """Average pooling layer. It computes average value of all features in pooling window.

        Args:
            pool_size (tuple): 2-d tuple. Adjust size of window pooling will hold in.
        """
        self.pool_size = pool_size
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.x = None
        self.der_w = None
        self.der_b = None
        self.type = 'AveragePooling_' + '%dx%d' % (pool_size[0], pool_size[1])

    def ForwardProp(self, inputs):
        self.x = inputs
        self.y = np.zeros((*np.shape(self.x)[:2], np.shape(self.x)[2] // self.pool_size[0], np.shape(self.x)[3] // self.pool_size[1]), dtype='float64')
        for row in range(np.shape(self.y)[2]):
            for col in range(np.shape(self.y)[3]):
                self.y[:, :, row, col] = np.average(
                    self.x[
                        :,
                        :,
                        row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                        col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                    ],
                    axis=(2, 3)
                )
    def BackProp(self, der_w, der_b, jac):
        self.der_w = der_w
        self.der_b = der_b
        self.jac_to_go = np.zeros_like(self.x, dtype='float64')
        for row in range(np.shape(jac)[2]):
            for col in range(np.shape(jac)[3]):
                self.jac_to_go = np.transpose(self.jac_to_go, (2, 3, 0, 1))
                self.jac_to_go[
                    row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1],
                    :,
                    :
                ] = jac[:, :, row, col] / (self.pool_size[0] * self.pool_size[1])
                self.jac_to_go = np.transpose(self.jac_to_go, (2, 3, 0, 1))

class Dropout(Layer):
    def __init__(self, keep_prob):
        """Dropout layer. It sets to zero outputs of layer with certain porbability.

        Args:
            keep_prob (float): Probability of keeping single output in initial state.
        """
        self.keep_prob = keep_prob
        self.y = None
        self.jac_to_go = None
        self.der_w = None
        self.der_b = None
        self.type = 'Dropout' + str(self.keep_prob)

    def ForwardProp(self, inputs):
        self.dy = np.random.uniform(0, 1, size=np.shape(inputs)) <=self.keep_prob
        self.y = inputs * self.dy

    def BackProp(self, der_w, der_b, jac):
        self.jac_to_go = jac * self.dy
        self.der_w = der_w
        self.der_b = der_b



class Convolution2d(Layer):
    def __init__(self, kernel_dim, padding):
        """2-d convolution layer.

        Args:
            kernel_dim (iterable): Size of weights matrix.
            First value - number of output channels of layer.
            Second and third values - numbers of rows and columns of kernel respectivly.
            padding (string): 'SAME' if you want feature map doesn't change it's size and 'VALID' otherwise.
        """
        assert not ((kernel_dim[1] % 2 == 0 or kernel_dim[2] % 2 == 0) and padding == 'SAME'), 'Kernel with SAME ' \
                                                                                               'padding ' \
                                                                                            'type must have middle cell'
        self.channels = kernel_dim[0]
        self.out_dim = self.channels
        self.use_bias = False
        self.padding = padding
        self.kernel_dim = list(kernel_dim)  # (channels_in, channels_out, row, column)
        self.x = None
        self.w = None
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.shape_w = None
        self.der_w = None
        self.der_b = None
        self.type = 'Convolution2D_' + '%dx%dx%d' % (kernel_dim[0], kernel_dim[1], kernel_dim[2])

    @staticmethod
    def conv(feature_map, kernel):
        f_shape = np.shape(feature_map)
        k_shape = np.shape(kernel)
        output_rows = (f_shape[2] - k_shape[2]) + 1
        output_cols = (f_shape[3] - k_shape[3]) + 1
        output = np.zeros((f_shape[0], k_shape[1], output_rows, output_cols), dtype='float64')
        for chn in range(np.shape(output)[1]):
            for row in range(output_rows):
                for col in range(output_cols):
                    output[:, chn, row, col] = np.sum(feature_map[
                        :,
                        :,
                        row:row+k_shape[2],
                        col:col+k_shape[3]
                    ] * kernel[:, chn, :, :], (1, 2, 3))
        return output

    def ForwardProp(self, inputs):
        self.x = inputs
        if self.padding == 'SAME':
            row_frame = ((np.shape(self.x)[2] - 1) + self.kernel_dim[2] - np.shape(self.x)[2]) // 2
            col_frame = ((np.shape(self.x)[3] - 1) + self.kernel_dim[3] - np.shape(self.x)[3]) // 2
            shape = (np.shape(self.x)[0], np.shape(self.x)[1], np.shape(self.x)[2] + row_frame*2, np.shape(self.x)[3] + col_frame*2)
            _x = np.zeros(shape, dtype='float64')
            _x[:, :, row_frame:-row_frame, col_frame:-col_frame] += self.x
            self.x = _x
        self.y = self.conv(self.x, self.w)

    def BackProp(self, der_w, der_b, jac):  # jac - (batch_size, channels, rows, columns)
        df_dw = np.zeros((np.shape(self.y)[0], np.shape(self.w)[0], np.shape(self.w)[1], np.shape(self.w)[2], np.shape(self.w)[3]), dtype='float64')
        for chn in range(np.shape(self.w)[0]):
            for chn_k in range(np.shape(self.w)[1]):
                for row in range(np.shape(self.w)[2]):
                    for col in range(np.shape(self.w)[3]):
                        df_dw[:, chn, chn_k, row, col] += np.sum(self.x[
                            :,
                            chn,
                            row:row+np.shape(jac)[2],
                            col:col+np.shape(jac)[3]
                        ] * jac[:, chn_k, :, :], (1, 2))
        self.jac_to_go = np.zeros_like(self.x, dtype='float64')
        conv_w = self.w[:, :, ::-1, ::-1]
        row_frame = np.shape(self.w)[2]-1
        col_frame = np.shape(self.w)[3]-1
        conv_jac = np.zeros((*np.shape(jac)[:2], np.shape(jac)[2]+row_frame*2, np.shape(jac)[3]+col_frame*2))
        conv_jac[:, :, row_frame:-row_frame, col_frame:-col_frame] += jac
        for chn in range(np.shape(self.x)[1]):
            for row in range(np.shape(self.x)[2]):
                for col in range(np.shape(self.x)[3]):
                    for chn_k in range(np.shape(self.w)[1]):
                        self.jac_to_go[:, chn, row, col] = np.sum(conv_jac[
                            :,
                            chn_k,
                            row:row+np.shape(self.w)[2],
                            col:col+np.shape(self.w)[3],
                        ] * conv_w[chn, chn_k, :, :], (1, 2))
        der_w.append(np.reshape(np.average(df_dw, axis=0), -1))
        self.der_w = der_w
        self.der_b = der_b

    def Initializer(self, *args):
        self.kernel_dim.insert(0, args[0].channels)
        self.w = np.random.uniform(-1, 1, size=self.kernel_dim)
        self.shape_w = np.shape(self.w)


class Input(Layer):
    def __init__(self, input_dim):
        """Initial layer of every network.

        Args:
            input_dim (int/iterable): Number of elements of input vector or size of input matrix with number of channels at first position.
        """
        self.out_dim = input_dim
        self.y = None
        self.der_w = None
        self.der_b = None
        if isinstance(self.out_dim, int):
            self.channels = 1
        else:
            self.channels = self.out_dim[0]
        self.type = 'Input' + str(self.out_dim)

    def ForwardProp(self, inputs):
        if isinstance(self.out_dim, int):
            self.y = inputs[:, :, np.newaxis, np.newaxis]
        else:
            self.y = inputs

    def BackProp(self, der_w, der_b, jac):
        self.der_w = der_w
        self.der_b = der_b


class Output(Layer):
    def __init__(self, out_dim):
        self.in_dim = out_dim
        self.out_dim = 1
        self.der_w = []
        self.der_b = []
        self.y = None
        self.jac_to_go = None
        self.type = 'Output' + str(self.in_dim)

    def ForwardProp(self, inputs):
        self.y = inputs

    def BackProp(self, jac):
        self.jac_to_go = jac
        self.der_w = []
        self.der_b = []


class Model():
    def __init__(self):
        """Your model.
        """
        self.model = []
        self.loss_dic = {
            'binary_crossentropy': BinaryCrossEntropy()
        }
        self.loss_func = None
        self.der_w = []  # Resulting weights gradient vector for whole batch
        self.der_b = []  # Resulting biases gradient vector for whole batch
        self.grad_w = 5  # Weights gradient vector for loss function
        self.grad_b = 5  # Biases gradient vector for loss function
        self.grad = 5  # Common gradient vector for loss function
        self.max_iter = None  # Maximum number of iteration during trainig process
        self.loss = 100
        self.acc = 0
        self.model_output = None
        self.iter_start_time = 0
        self.start_time = 0
        self.iteration = 0
        self.type = None
        self.model_sign = ''
        self.callbacks = []
        self.chpoint_dir = None
        self.chpoint_filename = None
        self.chpoint_delay = None
        self.chpoint_max = None

    def add(self, layer):
        """Method to add layers into your model.

        Args:
            layer (Layer): Layer wich you wnat to add.
        """
        self.model.append(layer)

    def comp(self, optimizer, loss, init=None):
        """Method to compile your model.

        Args:
            optimizer (string): Type of optimizer. Now available: gradient_descent.
            loss (string): Type of loss function. Now availbale: binary_crossentropy.
            init (string): Initialization type of ALL weights of Dense layers in your model. Now available: uniform, xavier, xe.
        """
        init = str(init)
        self.model_comp = list(filter(lambda layer: not isinstance(layer, (Activation, MaxPooling, AveragePooling, Dropout)), self.model))
        self.model.append(Output(self.model_comp[len(self.model_comp) - 1].out_dim))
        self.model_comp.append(self.model[len(self.model) - 1])
        for layer in range(1, len(self.model_comp) - 1):
            self.model_comp[layer].Initializer(self.model_comp[layer - 1], self.model_comp[layer + 1], init)
        self.loss_func = self.loss_dic[loss]
        self.optimimzer = Optimizer(optimizer, self.model)
        if any(map(lambda x: type(x) == Convolution2d, self.model)):
            self.type = 'ConvolutionNN'
        elif any(map(lambda x: type(x) == Dense, self.model)) and not any(map(lambda x: type(x) == Convolution2d, self.model)):
            self.type = 'FullyConnectedNN'
        else:
            self.type = 'UnknownTypeNN'
        for layer in self.model:
            self.model_sign += layer.type + '/'

    def fit(self, train_data, train_labels, val_data, early_stopping=lambda *args: True, batch_size=1, max_iter=100):
        """Method to fit your model.

        Args:
            train_data (ndarray): Train data.
            train_labels (ndarray): Train lables.
            val_data (tuple): Tuple of your validation data wich contains features and lables respectively.
            early_stopping (callable, optional): Early stop function.
            batch_size (int, optional): Batch size. Defaults to 1.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        """
        self.start_time = datetime.datetime.now()
        self.max_iter = max_iter
        while self.iteration <= self.max_iter and early_stopping(val_data, self.model, self.loss_func, self.max_iter, (self.start_time, self.iter_start_time)):
            self.iter_start_time = datetime.datetime.now()
            # Batch sampling
            indices = np.random.choice(np.array(range(np.shape(train_data)[0])), size=batch_size, replace=False)
            batch_data = train_data[indices]
            batch_labels = train_labels[indices]
            # Forward propagation
            self.exploit(batch_data)
            # Back propagation
            self.model[len(self.model)-1].BackProp(self.loss_func.df(self.model_output, batch_labels))
            for i in range(len(self.model) - 2, -1, -1):
                self.model[i].BackProp(
                    self.model[i + 1].der_w,
                    self.model[i + 1].der_b,
                    self.model[i + 1].jac_to_go
                )
            self.grad_w = np.array(self.model[i].der_w)
            self.grad_b = np.array(self.model[i].der_b)
            self.model = self.optimimzer(jac_w=self.grad_w, jac_b=self.grad_b, step=1)
            self.der_w.clear()
            self.der_b.clear()
            self.iteration += 1
            for call in self.callbacks:
                call()


    def exploit(self, data):
        """Compute model output on data.

        Args:
            data (ndarray): Features, input of the model.
        """
        self.model[0].ForwardProp(data)
        for i in range(1, len(self.model)):
            self.model[i].ForwardProp(self.model[i - 1].y)
        self.model_output = self.model[len(self.model) - 1].y

        return self.model_output

    def AddCallback(self, callback):
        """Add callback function which is executed after each iteration.

        Args:
            callback (callable): Callback.
        """
        self.callbacks.append(callback)


class EarlyStopping(Model):
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=False, mode='auto', delay=1):
        """Imbedded early stop callback.

        Args:
            monitor (str, optional): Quantity to be monitored. Defaults to 'val_loss'.
            min_delta (int, optional): Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement. Defaults to 0.
            patience (int, optional): Number of iterations with no improvement after which training will be stopped. Defaults to 0.
            verbose (bool, optional): Verbosity flag. Defaults to False.
            mode (str, optional): One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing;
            in "max" mode it will stop when the quantity monitored has stopped increasing;
            in "auto" mode, the direction is automatically inferred from the name of the monitored quantity. Defaults to 'auto'.
            delay (int, optional): Delay between checking changes of monitoring value. Defaults to 1.
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.delay = delay
        self.mon_value = None
        self.full_mon_value = None
        self.prev_mon_value = None
        self.average_mon_value = 0
        self.iter_to_abort = self.patience
        self.iteration = -1
        self.is_changing = True
        self.speed = 0
        self.speed_now = None
        assert not ((self.monitor != 'val_loss' and self.monitor != 'val_acc') or \
            (self.mode != 'auto' and self.mode != 'min' and self.mode != 'max')), 'Wrong value.'
        if (self.mode == 'auto' and self.monitor == 'val_loss') or self.mode == 'min':
            self.changing_dir = 1
            self.full_mon_value = 'Loss on validation data'
            self.prev_mon_value = 1000
        else:
            self.changing_dir = -1
            self.full_mon_value = 'Accuracy on validation data'
            self.prev_mon_value = 0

    def __call__(self, val_data, *args):  # args[0] - model; args[1] - loss_func; args[2] - max_iter; args[3] - (start_time, iter_start_time)
        if self.iteration == -1:
            self.iteration += 1
            self.loss_func = args[1]
            self.max_iter = args[2]
            return True
        self.start_time = args[3][0]
        self.iter_start_time = args[3][1]
        self.model = args[0]
        self.iteration += 1
        self.exploit(val_data[0])
        self.loss_func.results(self.model_output, val_data[1])
        if self.monitor == 'val_loss':
            self.mon_value = round(self.loss_func.f(self.model[len(self.model) - 1].y, val_data[1]), 4)
        else:
            self.mon_value = round(self.loss_func.accuracy(), 4)
        self.average_mon_value += self.mon_value / self.delay
        if self.verbose:
            return self._verbose_is_changing()
        else:
            return self._is_changing()

    def _verbose_is_changing(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print('Iteration %d/%d:' % (self.iteration, self.max_iter))
        print('%s: %f' % (self.full_mon_value, self.mon_value))
        res = self._is_changing()
        if self.iteration % self.delay == 0:
            print('Is changing: ' + str(self.is_changing))
            if not self.is_changing:
                print('Iterations to stop left: %d' % (self.iter_to_abort))
            else:
                print('Iterations to stop left: N/A')
        else:
            print('Is changing: /%s/ *Not updated* Update after %d rechecks' % (str(self.is_changing), self.delay-(self.iteration % self.delay)))
            if self.iter_to_abort == self.patience:
                print('Iterations to stop left: N/A')
            else:
                print('Iterations to stop left: %d' % (self.iter_to_abort))
        self.progress_bar()
        return res

    def _is_changing(self):
        if self.iteration % self.delay == 0:
            self.is_changing = (self.prev_mon_value - self.average_mon_value) * self.changing_dir >= self.min_delta
            if not self.is_changing and self.iter_to_abort != 0:
                self.iter_to_abort -= 1
                self.average_mon_value = 0
                return True
            elif not self.is_changing and self.iter_to_abort == 0:
                return False
            else:
                self.iter_to_abort = self.patience
                self.prev_mon_value = self.average_mon_value
                self.average_mon_value = 0
                return True
        else:
            return True

    def progress_bar(self):
        self.iter_time = datetime.datetime.now() - self.iter_start_time
        self.iter_time_int = self.iter_time.seconds + self.iter_time.microseconds * 10**(-6)
        try:
            self.iter_time_int += self.iter_time.minutes * 60
            self.iter_time_int += self.iter_time.hours * 3600
        except Exception:
            pass
        self.speed_now = 1 / self.iter_time_int
        self.percentage = round(self.iteration / self.max_iter * 100)
        self.speed = self.speed_now * 0.9 + self.speed * 0.1
        self.time_left = str(datetime.timedelta(seconds=(self.max_iter - self.iteration) / self.speed)).split('.')[0]
        self.time_pass = str(datetime.datetime.now() - self.start_time).split('.')[0]
        percent_string = ' '*49 + str(self.percentage) + '%'
        print(percent_string)
        print('[' + '='*self.percentage + ' '*(100-self.percentage) + ']')
        print('Estimated time to reach max_iter point: ' + str(self.time_left))
        print('Time passed: ' + str(self.time_pass))


class ModelCheckpointMgr(Model):
    def __init__(self, model, work_dir='checkpoints/'):
        """Manager of your model checkpoints.

        Args:
            model (Model): Your model.
            work_dir (str, optional): Directory with checkpoint files. Defaults to 'checkpoints/'.
        """
        self.model = model
        self.iteration = 0
        self.type = None
        self.chpoint_filename = None
        if work_dir[len(work_dir)-1] != '/':
            self.chpoint_dir = work_dir + '/'
        else:
            self.chpoint_dir = work_dir

    def MakeCheckpoint(self):
        """Callback which creates checkpoints. Add it via AddCallback to your model.
        """
        if self.model.iteration % self.model.chpoint_delay == 0:
            list_dir = os.listdir(self.model.chpoint_dir)
            if len(list(filter(lambda x: x.startswith(self.model.type), list_dir))) == self.model.chpoint_max:
                list_dir = sorted(list_dir)
                os.remove(self.chpoint_dir+list_dir.pop(0))
            self.model.chpoint_filename = self.model.type + str(datetime.datetime.now()).replace(':', '')[11:].split('.')[0] + '.txt'
            with open(self.model.chpoint_dir+self.model.chpoint_filename, 'w') as f:
                for layer in self.model.model:
                    f.write(layer.type + '/')
                f.write('\n')
                for layer in self.model.model_comp[1:len(self.model.model_comp)-1]:
                    w = layer.w
                    w = np.reshape(layer.w, -1)
                    for weight in w:
                        f.write(str(weight) + ',')
                    f.write('\n')

    def LoadWeights(self, filename):
        """Load weights to your model from checkpoint file.

        Args:
            filename (string): Checkpoint file.
        """
        with open(self.chpoint_dir+filename, 'r') as f:
            assert self.model.model_sign == f.readline()[:-1], 'Model signatures do not match'
            for lay_i in range(1, len(self.model.model_comp)-1):
                weights = f.readline().split(',')
                weights.pop()
                weights = np.reshape(np.array(weights, dtype='float64'), np.shape(self.model.model_comp[lay_i].w))
                self.model.model_comp[lay_i].w = weights

    def EnableCheckpoints(self, chpoint_delay=10, chpoint_max=3):
        """Enable checkpoints.

        Args:
            chpoint_delay (int, optional): Delay of checkpoint creation. Defaults to 10.
            chpoint_max (int, optional): Max value of checkpoint files. Defaults to 3.
        """
        self.model.chpoint_delay = chpoint_delay
        self.model.chpoint_max = chpoint_max
        self.model.chpoint_dir =self.chpoint_dir
