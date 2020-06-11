import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from PIL import Image, ImageDraw
import sys
import random
#import keras

def read_label(f_name):
    """
    Read a "FishPose" label in the form: id y   x
    """
    f = open(f_name)
    label = np.zeros((10,3))
    l_counter = 0
    for line in f:
        things = line.strip().split('\t')
        label[l_counter,:] = list(map(float, things))
        l_counter += 1
    f.close()
    return label

def load_dataset(folder):
    fil_list = os.listdir(folder)
    n_f = len(fil_list)//2
    X_orig = np.zeros((n_f, 100, 300, 1))
    Y_orig = np.zeros((n_f, 10, 3))
    counter = 0
    for f in fil_list:
        if f.endswith(".txt"):
            img_f = f[:-3] + "bmp"
            X_orig[counter,:,:,0] = np.array(Image.open(folder+img_f))
            Y_orig[counter,:,:] = read_label(folder+f) 
            counter += 1

    X = X_orig/255.

    return X, Y_orig

def FP_loss(y_pred, y_true):
    pres_weight = 0.05

    pres_true = y_true[:,:,0]
    coord_true = y_true[:,:,1:]
    pred = tf.reshape(y_pred, (-1,10,3))

    pres_pred = pred[:,:,0]
    coord_pred = pred[:,:,1:]

    pres_cost = tf.nn.sigmoid_cross_entropy_with_logits(labels = pres_true, logits = pres_pred)
    coord_cost = tf.norm(tf.multiply(coord_pred - coord_true,[1,3]), ord = 2, axis=2)

    pres_not_ignore = tf.math.sign(pres_true + 1)
    pres_inds = tf.nn.relu(pres_true)

    RealPresCost = tf.math.reduce_mean(tf.math.multiply(pres_not_ignore, pres_cost)) * pres_weight#/ni_sum
    RealCoordCost = tf.math.reduce_mean(tf.math.multiply(pres_inds, coord_cost))#/pi_sum
    cost = RealPresCost + RealCoordCost
    
    return cost, RealPresCost, RealCoordCost

def create_placeholders():
    X = tf.keras.backend.placeholder(dtype = tf.float32, shape = (None, 100, 300, 1))
    Y = tf.keras.backend.placeholder(dtype = tf.float32, shape = (None, 10, 3))
    return X, Y

def initialize_parameters():
    tf.compat.v1.set_random_seed(1)
    W1 = tf.compat.v1.get_variable("W1", [5,5,1,48], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.compat.v1.get_variable("W2", [5,5,48,128], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.compat.v1.get_variable("W3", [5,5,128,192], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.compat.v1.get_variable("W4", [1,1,192,64], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W5 = tf.compat.v1.get_variable("W5", [3,3,192,192], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1, "W2":W2, "W3":W3, "W4":W4, "W5":W5}
    return parameters

def forward_prop(X, P):
    #P is for parameters
    Z1 = tf.nn.conv2d(X, P['W1'], strides = [1,2,2,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool2d(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1, P['W2'], strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool2d(A2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    Z3 = tf.nn.conv2d(P2, P['W3'], strides = [1,2,2,1], padding = 'SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool2d(A3, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

    Z4 = tf.nn.conv2d(P3, P['W4'], strides = [1,1,1,1], padding = 'SAME')
    A4 = tf.nn.relu(Z4)
    #Z4 = tf.nn.conv2d(P3, P['W4'], padding = 'VALID')
    #A4 = tf.nn.relu(Z4)
    #Z5 = tf.nn.conv2d(A3, P['W5'], strides = [1,1,1,1], padding = 'SAME')
    #A5 = tf.nn.relu(Z5)
    #P5 = tf.nn.max_pool2d(A2, ksize = [1,3,3,1], strides = [1,3,3,1], padding = 'SAME')

    F6 = tf.contrib.layers.flatten(A4)
    #A6 = tf.contrib.layers.fully_connected(F6, 512)

    A7 = tf.contrib.layers.fully_connected(F6, 512)

    Z8 = tf.contrib.layers.fully_connected(A7, 30, activation_fn = None)

    return Z8

def FP_loss_assess(y_pred, y_true):
    return y_pred

def batch_generator(X, Y, seed, b_size = 64):
    np.random.seed(seed)
    m = X.shape[0]
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    X_shuffled = X[permutation, :, :, :]
    Y_shuffled = Y[permutation, :, :]

    batch_num = m//b_size
    Batches = []
    for b in range(batch_num):
        Batches.append((X_shuffled[b*b_size:(b+1)*b_size], Y_shuffled[b*b_size:(b+1)*b_size]))

    Batches.append((X_shuffled[batch_num*b_size:], Y_shuffled[batch_num*b_size:]))
    return Batches        

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00001, num_epochs = 1500, batch_size = 64, print_cost = True):
    ops.reset_default_graph()
    tf.compat.v1.set_random_seed(1)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders()
    parameters = initialize_parameters()
    Z8 = forward_prop(X, parameters)
    cost_ph = FP_loss(Z8, Y)

    global_step = tf.Variable(0, trainable=False)
    tf.compat.v1.train.exponential_decay(learning_rate, global_step, 100, 0.96)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost_ph[2], global_step = global_step)

    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            Batches = batch_generator(X_train, Y_train, epoch, b_size=batch_size)
            for minibatch in Batches:
                _, cost = sess.run(fetches = [optimizer, cost_ph], feed_dict = {X: minibatch[0], Y: minibatch[1]})
                costs.append(cost[0])
                
            if print_cost and epoch%10 == 9:
                _, pres, coord = FP_loss(Z8, Y)
                test_pres = pres.eval({X: X_test, Y: Y_test})
                test_coord = coord.eval({X: X_test, Y: Y_test})
                print("%i, %f, %f, %f, %f" % (epoch, cost[1], cost[2], test_pres, test_coord))

        assessment = FP_loss_assess(Z8, Y)
        y_pred = assessment.eval({X: X_test, Y: Y_test})

    return parameters, y_pred

def random_color_gen():
    # Generates a random color in hex format
    color = "#"+''.join([random.choice('6789ABCDEF') for j in range(6)])
    #emmitted lower values to make color brighter
    return color

def procure_output(X_test, Y_test, y_pred, out_folder):
    r = [1,1]
    out = open(out_folder + 'output.txt', 'w')

    for i in range(X_test.shape[0]):
        img = Image.fromarray((X_test[i,:,:,0] * 255.).astype(np.uint8), 'L')
        img = img.convert('RGB')
        draw = ImageDraw.Draw(img)
        lbl = (Y_test[i,:,:] + [0, 0.5, 0.5]) * [1,100,300]
        pred = (y_pred[i,:].reshape(-1, 3) + [0, 0.5, 0.5]) * [1,100, 300]

        lbl = lbl[:,[0,2,1]]
        pred = pred[:,[0,2,1]]
          
        random.seed(0)
        for d in range(lbl.shape[0]):
            if lbl[d,0] == 1:
                start = tuple(lbl[d,1:])
                s0 = tuple(lbl[d,1:] - r)
                s1 = tuple(lbl[d,1:] + r)
                end = tuple(pred[d,1:])
                color = random_color_gen()
                draw.line([start, end], fill = color)
                draw.rectangle([s0, s1], fill = color)
                
                toWrite = list(map(str, [i, lbl[d, 1], lbl[d, 2], pred[d, 1], pred[d, 2]]))
                out.write('\t'.join(toWrite) + '\n')

        img.save(out_folder + str(i) + '.bmp', 'BMP')
    out.close()
                
            
        

        


















    """
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
    """
