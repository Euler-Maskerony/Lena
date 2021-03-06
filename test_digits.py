import numpy as np
import mnist
from Lena import Activation, Dense, Dropout, Model, Convolution2d, MaxPooling, AveragePooling, Input, EarlyStopping, ModelCheckpointMgr

x_test = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', 'data/')
y_test = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', 'data/')
x_test= x_test[:, np.newaxis, :, :]/255

model = Model()
model.add(Input((1, 28, 28)))
model.add(Convolution2d([10, 5, 5], padding='SAME'))
model.add(Activation('sigmoid'))
model.add(MaxPooling((2, 2)))
model.add(Convolution2d([15, 5, 5], padding='SAME'))
model.add(Activation('sigmoid'))
model.add(MaxPooling((2, 2)))
model.add(Convolution2d([20, 6, 6], padding='VALID'))
model.add(Activation('relu'))
model.add(MaxPooling((2, 2)))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.comp('gradient_descent', 'binary_crossentropy')

chpoint_mgr = ModelCheckpointMgr(model)
chpoint_mgr.LoadWeights('ConvolutionNN002006.txt')

def digit_rec(data):
    return int(np.argmax(model.exploit(data), axis=1))

if __name__ == '__main__':
    output = model.exploit(x_test)
    accuracy = np.average(np.argmax(np.reshape(output, (np.shape(x_test)[0], -1)), axis=1) == y_test)

    print(accuracy)
