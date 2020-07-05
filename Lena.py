# Lena - Learning and assEmbly tool for NeurAl networks

import numpy as np
from math import e, ceil


def ext_dot(a, b):
    res = np.zeros((a.shape[0], a.shape[1], b.shape[1]))
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


class Loss:  # TODO Rewrite/Get rid of it somehow
    pass


class BinaryCrossEntropy(Loss):
    @staticmethod
    def f(x, y):
        res = np.zeros_like(x)
        x += 10**(-10)
        for batch_el in range(np.shape(res)[0]):
            res[batch_el] = -y[batch_el] * np.log(x[batch_el]) - (1 - y[batch_el]) * np.log(1 - x[batch_el])
            res[batch_el] = np.sum(res[batch_el])
        res = np.average(res)
        res_x = np.argmax(np.reshape(x, (np.shape(x)[0], -1)), axis=1)
        res_y = np.argmax(np.reshape(y, (np.shape(y)[0], -1)), axis=1)
        acc = np.zeros(np.shape(res_x)[0], dtype=bool)
        for batch_el in range(np.shape(res_x)[0]):
            acc[batch_el] = np.array_equal(res_x[batch_el], res_y[batch_el])

        print(res_x)
        print(res_y)
        print('Accuracy:')
        print(np.average(acc))
        return res

    @staticmethod
    def df(x, y):
        res = np.zeros_like(x)
        x += 10**(-10)
        for batch_el in range(np.shape(res)[0]):
            res[batch_el] = (-y[batch_el] / (x[batch_el] + 10**(-7)) + (1 - y[batch_el]) / (1 - x[batch_el] - 10**(-7))) / np.size(res[batch_el])
        return res


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


class Layer:
    initializer_dic = {
        'uniform': Initializer.uniform,
        'xavier': Initializer.xavier,
        'xe': Initializer.xe
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

    def ForwardProp(self, inputs):
        self.x = np.reshape(inputs, (np.shape(inputs)[0], 1, 1, -1))  # (batch_size; 1; 1; input)
        self.y = np.transpose(np.dot(self.w, np.transpose(self.x, (0, 1, 3, 2))), (1, 2, 3, 0)) + self.b * int(self.use_bias)
        self.dy_dw = np.transpose(np.ones((np.shape(self.w)[0], np.shape(self.x)[0], np.shape(self.w)[1])) * self.x[:, 0, 0, :], (1, 0, 2))
        self.dy_db = np.ones((np.shape(self.x)[0], np.shape(self.b)[0])) * int(self.use_bias)

    def BackProp(self, der_w, der_b, jac):
        jac = jac[:, 0, 0, :]
        df_dw = np.reshape(np.average(np.transpose(np.transpose(self.dy_dw, (2, 0, 1)) * jac, (1, 2, 0)), axis=0), -1)
        df_db = np.reshape(np.average(self.dy_db * jac, axis=0), -1)
        der_w.append(df_dw)
        der_b.append(df_db)
        self.der_w = der_w
        self.der_b = der_b
        self.jac_to_go = np.transpose(np.dot(np.transpose(self.w), np.transpose(jac)))
        self.jac_to_go = self.jac_to_go[:, np.newaxis, np.newaxis, :]

    def Initializer(self, prev_lay, next_lay, init_type):
        init = super().initializer_dic[init_type]
        self.in_dim = prev_lay.out_dim
        self.w = np.zeros((self.out_dim, self.in_dim))
        self.b = np.zeros(self.out_dim)
        self.shape_w = np.shape(self.w)
        self.w = init(self.shape_w, next_lay.out_dim)


class MaxPooling(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.x = None
        self.der_w = None
        self.der_b = None

    def ForwardProp(self, inputs):
        self.x = inputs
        self.y = np.zeros((*np.shape(self.x)[:2], np.shape(self.x)[2] // self.pool_size[0], np.shape(self.x)[3] // self.pool_size[1]))
        self.dy = np.zeros(np.shape(self.x))
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
        self.jac_to_go = np.zeros_like(self.x)
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
                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                ], (2,3,0,1)) * jac[:, :, row, col], (2,3,0,1))


class AveragePooling(Layer):
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.x = None
        self.der_w = None
        self.der_b = None

    def ForwardProp(self, inputs):
        self.x = inputs
        self.y = np.zeros((*np.shape(self.x)[:2], np.shape(self.x)[2] // self.pool_size[0], np.shape(self.x)[3] // self.pool_size[1]))
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
        self.jac_to_go = np.zeros_like(self.x)
        for row in range(np.shape(jac)[2]):
            for col in range(np.shape(jac)[3]):
                self.jac_to_go[
                    :,
                    :,
                    row*self.pool_size[0]:row*self.pool_size[0]+self.pool_size[0],
                    col*self.pool_size[1]:col*self.pool_size[1]+self.pool_size[1]
                ] = jac[:, :, row, col] / (self.pool_size[0] * self.pool_size[1])

class Dropout(Layer):
    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.y = None
        self.jac_to_go = None
        self.der_w = None
        self.der_b = None

    def ForwardProp(self, inputs):
        self.dy = np.random.uniform(0, 1, size=np.shape(inputs)) <=self.keep_prob
        self.y = inputs * self.dy

    def BackProp(self, der_w, der_b, jac):
        self.jac_to_go = jac * self.dy
        self.der_w = der_w
        self.der_b = der_b



class Convolution2d(Layer):
    def __init__(self, kernel_dim, strides, padding):
        assert not ((kernel_dim[1] % 2 == 0 or kernel_dim[2] % 2 == 0) and padding == 'SAME'), 'Kernel with SAME ' \
                                                                                               'padding ' \
                                                                                            'type must have middle cell'
        self.channels = kernel_dim[0]
        self.out_dim = self.channels
        self.use_bias = False
        self.padding = padding
        self.strides = strides  # (batch_size, channels, rows, columns)
        self.kernel_dim = list(kernel_dim)  # (channels_in, channels_out, row, column)
        self.x = None
        self.w = None
        self.y = None
        self.dy = None
        self.jac_to_go = None
        self.shape_w = None
        self.der_w = []
        self.der_b = []

    @staticmethod
    def conv(feature_map, kernel, strides):
        f_shape = np.shape(feature_map)
        k_shape = np.shape(kernel)
        output_rows = (f_shape[2] - k_shape[2]) // strides[2] + 1
        output_cols = (f_shape[3] - k_shape[3]) // strides[3] + 1
        output = np.zeros((f_shape[0], k_shape[1], output_rows, output_cols))
        dout_dw = np.zeros((f_shape[0], k_shape[1], k_shape[0], k_shape[2], k_shape[3], output_rows, output_cols))
        for batch_el in range(0, np.shape(output)[0], strides[0]):
            for row in range(0, output_rows, strides[2]):
                for col in range(0, output_cols, strides[3]):
                    for chn in range(0, np.shape(output)[1], strides[1]):
                        output[batch_el, chn, row, col] = np.sum(feature_map[batch_el, :, row:row + k_shape[2], col:col + k_shape[3]] * kernel[:, chn, :, :])
                        dout_dw[batch_el, chn, :, :, :, row, col] = feature_map[batch_el, :, row:row + k_shape[2], col:col + k_shape[3]]

        return output, dout_dw

    def ForwardProp(self, inputs):
        self.x = np.array(inputs)  # TODO Get it back
        if self.padding == 'SAME':
            row_frame = (self.kernel_dim[2] - 1) // 2
            col_frame = (self.kernel_dim[3] - 1) // 2
            shape = (np.shape(self.x)[0], np.shape(self.x)[1], np.shape(self.x)[2] + row_frame * 2,
                     np.shape(self.x)[3] + col_frame * 2)
            _x = np.zeros(shape)
            _x[:, :, row_frame:shape[2]-row_frame, col_frame:shape[3]-col_frame] += self.x
            self.x = _x
        #assert (np.shape(self.x)[2] - self.kernel_dim[1]) % self.strides[0] != 0 or \
        #       (np.shape(self.x)[3] - self.kernel_dim[2]) % self.strides[1] != 0, \
        #    'Invalid strides value'
        self.y, self.dy = self.conv(self.x, self.w, self.strides)

    def BackProp(self, der_w, der_b, jac):  # jac -> (batch_size, channels, rows, columns)
        jac = np.reshape(jac, np.shape(self.y))
        df_dw = np.zeros((np.shape(self.y)[0], np.shape(self.w)[0], np.shape(self.w)[1], np.shape(self.w)[2], np.shape(self.w)[3]))
        for batch_el in range(0, np.shape(self.y)[0]):
            for chn in range(0, np.shape(self.y)[1]):
                for row in range(0, np.shape(self.y)[2]):
                    for col in range(0, np.shape(self.y)[3]):
                        df_dw[batch_el, :, chn, :, :] += self.dy[batch_el, chn, :, :, :, row, col] * jac[batch_el, chn, row, col]
        self.jac_to_go = np.zeros_like(self.x)
        for batch_el in range(np.shape(self.x)[0]):
            for chn in range(0, np.shape(self.x)[1]):
                for row in range(0, np.shape(self.x)[2]):
                    for col in range(0, np.shape(self.x)[3]):
                        for k_chn in range(np.shape(self.w)[1]):
                            for k_row in range(np.shape(self.w)[2]):
                                for k_col in range(np.shape(self.w)[3]):
                                    if (row - k_row >= 0) and (col - k_col >= 0):
                                        y_row = max(ceil((row - np.shape(self.w)[2]) / self.strides[2]), 0)
                                        y_col = max(ceil((col - np.shape(self.w)[3]) / self.strides[3]), 0)
                                        self.jac_to_go[batch_el, chn, row, col] += self.w[chn, k_chn, k_row, k_col] * jac[batch_el, k_chn, y_row, y_col]
        der_w.append(np.reshape(np.average(df_dw, axis=0), -1))
        self.der_w = der_w
        self.der_b = der_b

    def Initializer(self, *args):
        self.kernel_dim.insert(0, args[0].channels)
        self.w = np.zeros(tuple(self.kernel_dim))
        self.shape_w = np.shape(self.w)


class Input(Layer):
    def __init__(self, input_dim):
        self.out_dim = input_dim
        self.y = None
        self.der_w = None
        self.der_b = None
        if isinstance(self.out_dim, int):
            self.channels = 1
        else:
            self.channels = self.out_dim[0]

    def ForwardProp(self, inputs):
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

    def ForwardProp(self, inputs):
        self.y = inputs

    def BackProp(self, jac):
        self.jac_to_go = jac
        self.der_w = []
        self.der_b = []


class Model:
    def __init__(self):
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

    def add(self, layer):
        self.model.append(layer)

    def comp(self, optimizer, loss, init):
        model_comp = list(filter(lambda layer: not isinstance(layer, (Activation, MaxPooling, AveragePooling, Dropout)), self.model))
        self.model.append(Output(model_comp[len(model_comp) - 1].out_dim))
        model_comp.append(self.model[len(self.model) - 1])
        for lay in range(1, len(model_comp) - 1):
            model_comp[lay].Initializer(model_comp[lay - 1], model_comp[lay + 1], init)
        self.loss_func = self.loss_dic[loss]
        self.optimimzer = Optimizer(optimizer, self.model)

    def fit(self, train_data, train_labels, batch_size=4):
        if np.shape(train_data) == 2:
            train_data = train_data[:, np.newaxis, np.newaxis, :]
        while np.linalg.norm(self.grad) > 0.01:
            # Batch sampling
            indices = np.random.choice(np.array(range(np.shape(train_data)[0])), size=batch_size, replace=False)
            batch_data = train_data[indices]
            batch_labels = train_labels[indices]
            # Forward propagation
            self.model[0].ForwardProp(batch_data)
            for i in range(1, len(self.model)):
                self.model[i].ForwardProp(self.model[i - 1].y)
            # Back propagation
            self.model[i].BackProp(self.loss_func.df(self.model[i].y, batch_labels))
            for i in range(len(self.model) - 2, -1, -1):
                self.model[i].BackProp(
                    self.model[i + 1].der_w,
                    self.model[i + 1].der_b,
                    self.model[i + 1].jac_to_go
                )
            self.grad_w = np.array(self.model[i].der_w)
            self.grad_b = np.array(self.model[i].der_b)
            self.model = self.optimimzer(jac_w=self.grad_w, jac_b=self.grad_b, step=0.5)
            self.der_w.clear()
            self.der_b.clear()
            self.grad = np.hstack(tuple(self.grad_w))
            if self.grad_b.size > 0:
                self.grad = np.hstack((self.grad, np.hstack(tuple(self.grad_b))))
            print(self.loss_func.f(self.model[len(self.model) - 1].y, batch_labels))

        return self.loss_func.f(self.model[len(self.model) - 1].y, batch_labels)

if __name__ == "__main__":
    model = Model()
    model.add(Input((3, 15, 15)))
    model.add(Convolution2d([5, 3, 3], [1, 1, 1, 1], padding='SAME'))
    model.add(Activation('sigmoid'))
    model.add(MaxPooling((3, 3)))
    model.add(Convolution2d([8, 5, 5], [1, 1, 1, 1], padding='VALID'))
    model.add(Activation('tanh'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(0.78))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.comp('gradient_descent', 'binary_crossentropy', 'xavier')
    print(model.fit(np.random.uniform(0, 10, size=[10, 3, 15, 15]), np.random.uniform(0, 1, size=[10, 1]), batch_size=1))
