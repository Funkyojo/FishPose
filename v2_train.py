from Fishpose_util_v2 import *
import tensorflow as tf
import numpy as np

train_f = 'inputs/'
test_f = 'inputs/test_set/'

X_train, Y_train = load_dataset(train_f)
X_test, Y_test = load_dataset(test_f)

print('X_train, Y_train, X_test, Y_test shapes: ', X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

parameters, y_pred = model(X_train, Y_train, X_test, Y_test, num_epochs=1000, learning_rate=0.00002)
procure_output(X_test, Y_test, y_pred, 'output/')

