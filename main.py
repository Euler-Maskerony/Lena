import numpy as np
import mnist
from LENA import Activation, Dense, Dropout, Model, Convolution2d, MaxPooling, AveragePooling, Input

x_test = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', 'data/')
y_test = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', 'data/')
x_train = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz', 'data/')
y_train = mnist.download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', 'data/')
y_test, y_train = np.eye(10)[y_test], np.eye(10)[y_train]
x_test, x_train = np.reshape(x_test[:, np.newaxis, :, :]/255, (np.shape(x_test)[0], -1)), np.reshape(x_train[:, np.newaxis, :, :]/255, (np.shape(x_train)[0], -1))
print(np.shape(y_test))
print(np.shape(x_test))

model = Model()
model.add(Input(784))
model.add(Dense(784))
model.add(Activation('relu'))
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(0.78))
model.add(Dense(10))
model.add(Activation('softmax'))
model.comp('gradient_descent', 'binary_crossentropy', 'xavier')
print(model.fit(x_train, y_train, batch_size=100))
