import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image
import sys

def read_label(f_name):
    """
    Read a "FishPose" label in the form: id y   x
    """
    f = open(f_name)
    label = np.zeros((10,2)) #* np.nan
    for line in f:
        things = line.strip().split('\t')
        m_id, y, x = list(map(float, things))
        label[int(m_id),:] = [y, x]
    f.close()
    return label

def test_shuffle():
    A = np.random.randn(3,10,10,1)
    B = np.random.randn(3,10,2)
    A_S = np.copy(A)
    B_S = np.copy(B)
    np.random.seed(1)
    np.random.shuffle(A_S)
    np.random.seed(1)
    np.random.shuffle(B)
    
    return A, B, A_S, B_S
 

def load_dataset(folder):
    fil_list = os.listdir(folder)
    n_f = len(fil_list)//2
    X_train_orig = np.zeros((n_f, 100, 300, 1)) #* np.nan
    Y_train_orig = np.zeros((n_f, 10, 2)) #* np.nan
    counter = 0
    for f in fil_list:
        if f.endswith(".txt"):
            img_f = f[:-3] + "bmp"
            X_train_orig[counter,:,:,0] = np.array(Image.open(folder+img_f))
            Y_train_orig[counter,:,:] = read_label(folder+f) 
            counter += 1
    np.random.seed(1)
    np.random.shuffle(X_train_orig)
    np.random.seed(1)
    np.random.shuffle(Y_train_orig)

    X_test = X_train_orig[:10,:,:,:]/255.
    Y_test = Y_train_orig[:10,:,:] - [0.5, 0.5]
    X_train = X_train_orig[10:,:,:,:]/255.
    Y_train = Y_train_orig[10:,:,:] - [0.5, 0.5]
    classes = list(range(10))

    return X_train, Y_train, X_test, Y_test, classes

def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape = [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, shape = [None, n_y, 2])
    return X, Y

def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [11,11,1,48], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [5,5,48,128], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [3,3,128,192], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [3,3,192,192], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W5 = tf.get_variable("W5", [3,3,192,192], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2":W2, "W3":W3, "W4":W4, "W5":W5}
    return parameters

def forward_prop(X, P):
    #P is for parameters
    Z1 = tf.nn.conv2d(X, P['W1'], strides = [1,4,4,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, P['W2'], strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    Z3 = tf.nn.conv2d(P2, P['W3'], strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)

    #Z4 = tf.nn.conv2d(A3, P['W4'], strides = [1,1,1,1], padding = 'SAME')
    #A4 = tf.nn.relu(Z4)

    Z5 = tf.nn.conv2d(A3, P['W5'], strides = [1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A2, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    #F6 = tf.contrib.layers.flatten(P5)
    #A6 = tf.contrib.layers.fully_connected(F6, 2304)

    F7 = tf.contrib.layers.flatten(A5)
    A7 = tf.contrib.layers.fully_connected(F7, 2304)

    F8 = tf.contrib.layers.flatten(A7)
    Z8 = tf.contrib.layers.fully_connected(F8, 20, activation_fn = None)

    return Z8


def forward_prop_predict(X, P):
    #P is for parameters
    Z1 = tf.add(X, P['W1'], strides = [1,4,4,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    Z2 = tf.nn.conv2d(P1, P['W2'], strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    Z3 = tf.nn.conv2d(P2, P['W3'], strides = [1,1,1,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)

    #Z4 = tf.nn.conv2d(A3, P['W4'], strides = [1,1,1,1], padding = 'SAME')
    #A4 = tf.nn.relu(Z4)

    Z5 = tf.nn.conv2d(A3, P['W5'], strides = [1,1,1,1], padding = 'SAME')
    A5 = tf.nn.relu(Z5)
    P5 = tf.nn.max_pool(A2, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    #F6 = tf.contrib.layers.flatten(P5)
    #A6 = tf.contrib.layers.fully_connected(F6, 2304)

    F7 = tf.contrib.layers.flatten(A5)
    A7 = tf.contrib.layers.fully_connected(F7, 2304)

    F8 = tf.contrib.layers.flatten(A7)
    Z8 = tf.contrib.layers.fully_connected(F8, 20, activation_fn = None)

    return Z8


def compute_cost(Z8, Y):
    pred_coords = tf.reshape(Z8,(-1,10,2))
    cost = tf.reduce_mean(tf.norm(pred_coords - Y, ord = 2, axis=2))
    return cost, pred_coords

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00001, num_epochs = 1500, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z8 = forward_prop(X, parameters)
    cost = compute_cost(Z8, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost[0])

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            _, bla = sess.run(fetches = [optimizer, cost], feed_dict = {X: X_train, Y: Y_train})
            e_cost = bla[0]
            costs.append(e_cost)
            if print_cost and epoch%100 == 0:
                print("Cost after epoch %i: %f" % (epoch, e_cost))

    # Calculate the correct predictions
        
                accuracy, _ = compute_cost(Z8, Y)
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
                print("Valid cost: %f" % (test_accuracy))

    return parameters, bla[1]

def output_prediction(X, Y, P, m):
    assert(Y.shape == P.shape)
    folder = 'output/'
    for i in range(m):
        arr = X[i,:,:,0] * 255
        img = Image.fromarray(arr.astype(np.uint8), 'L')
        img.save(folder + str(i)+'.bmp', 'BMP')
        O = open(folder + str(i) + '_Y.txt', 'w')
        for k in range(Y.shape[1]):
            coords = [k, Y[i,k,0], Y[i,k,1]]
            coords = list(map(str, coords))
            O.write('\t'.join(coords) + '\n')
        O.close()
        pp = open(folder + str(i) + '_P.txt', 'w')
        for k in range(P.shape[1]):
            coords = [k, P[i,k,0], P[i,k,1]]
            coords = list(map(str, coords))
            pp.write('\t'.join(coords) + '\n')
        pp.close()

#normalize sets
folder = 'inputs/'
X_train, Y_train, X_test, Y_test, classes = load_dataset(folder)

"""
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
"""

parameters, pred_coords = model(X_train, Y_train, X_test, Y_test, num_epochs=1500, learning_rate=0.00005)
output_prediction(X_train, Y_train, pred_coords, 10)

"""
A,B,AS,BS = test_shuffle()
print(A[:,0,:,:])
print(B[:,0,:])
print(AS[:,0,:,:])
print(BS[:,0,:])
"""
