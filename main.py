import numpy as np
import mnist
from Lena import Activation, Dense, Dropout, Model, Convolution2d, MaxPooling, AveragePooling, Input, EarlyStopping, ModelCheckpointMgr

VAL_SIZE = 20

x_test = mnist.download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz', 'data/')
y_test = mnist.download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz', 'data/')
x_train = mnist.download_and_parse_mnist_file('train-images-idx3-ubyte.gz', 'data/')
y_train = mnist.download_and_parse_mnist_file('train-labels-idx1-ubyte.gz', 'data/')
y_test, y_train = np.eye(10)[y_test], np.eye(10)[y_train]
x_test, x_train = x_test[:, np.newaxis, :, :]/255, x_train[:, np.newaxis, :, :]/255
x_val, y_val = x_test[:VAL_SIZE], y_test[:VAL_SIZE]
x_test, y_test = x_test[VAL_SIZE:], y_test[VAL_SIZE:]

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=True, delay=3)

model = Model()
model.add(Input((1, 28, 28)))
model.add(Convolution2d([10, 5, 5], [1, 1, 1, 1], padding='SAME'))
model.add(Activation('sigmoid'))
model.add(MaxPooling((2, 2)))
model.add(Convolution2d([15, 5, 5], [1, 1, 1, 1], padding='SAME'))
model.add(Activation('sigmoid'))
model.add(MaxPooling((2, 2)))
model.add(Convolution2d([20, 6, 6], [1, 1, 1, 1], padding='VALID'))
model.add(Activation('relu'))
model.add(MaxPooling((2, 2)))
model.add(Dense(20))
model.add(Activation('sigmoid'))
model.add(Dropout(0.78))
model.add(Dense(10))
model.add(Activation('softmax'))
model.comp('gradient_descent', 'binary_crossentropy', 'xavier')

chpoint_mgr = ModelCheckpointMgr(model)
chpoint_mgr.EnableCheckpoints(chpoint_delay=10, chpoint_max=1)
chpoint_mgr.LoadWeights('ConvolutionNN172251.txt')
model.AddCallback(chpoint_mgr.MakeCheckpoint)

model.fit(x_train, y_train, (x_val, y_val), early_stopping=early_stop, batch_size=10, max_iter=500)
