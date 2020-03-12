import tensorflow as tf
import numpy as np
import sys
import time

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from dlADMM import common
from dlADMM.input_data import mnist, fashion_mnist


# initialize the neural network
def Net(images, label, num_of_neurons):
    seed_num = 13
    tf.random.set_seed(seed=seed_num)
    W1 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 28 * 28),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b1 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z1 = tf.matmul(W1, images) + b1
    a1 = common.relu(z1)
    tf.random.set_seed(seed=seed_num)
    W2 = tf.Variable(tf.random.normal(shape=(num_of_neurons, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b2 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z2 = tf.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    tf.random.set_seed(seed=seed_num)
    W3 = tf.Variable(tf.random.normal(shape=(10, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b3 = tf.Variable(tf.random.normal(shape=(10, 1),mean=0,stddev=0.1))
    imask = tf.equal(label, 0)
    z3 = tf.where(imask, -tf.ones_like(label), tf.ones_like(label))
    return W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3, images, labels):
    nums = labels.shape[1]
    z1 = tf.matmul(W1, images) + b1
    a1 = common.relu(z1)
    z2 = tf.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    z3 = tf.matmul(W3, a2) + b3
    cost = common.cross_entropy_with_softmax(labels, z3) / nums
    actual = tf.argmax(labels, axis=0)
    pred = tf.argmax(z3, axis=0)
    return (tf.reduce_sum(tf.cast(tf.equal(pred, actual),tf.float32)) / nums,cost)


# return the value of the augmented Lagrangian
def objective(x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, v1, v2, rho):
    r1 = tf.reduce_sum((z1 - tf.matmul(W1, x_train) - b1) * (z1 - tf.matmul(W1, x_train) - b1))
    r2 = tf.reduce_sum((z2 - tf.matmul(W2, a1) - b2) * (z2 - tf.matmul(W2, a1) - b2))
    r3 = tf.reduce_sum((z3 - tf.matmul(W3, a2) - b3) * (z3 - tf.matmul(W3, a2) - b3))
    loss = common.cross_entropy_with_softmax(y_train, z3)
    obj = loss + tf.linalg.trace(tf.matmul(z3 - tf.matmul(W3, a2) - b3, tf.transpose(u)))
    obj = obj + rho / 2 * r1 + rho / 2 * r2 + rho / 2 * r3
    obj = obj + rho / 2 * tf.reduce_sum((a1 - common.relu(z1) + v1) * (a1 - common.relu(z1) + v1)) + rho / 2 * tf.reduce_sum(
        (a2 - common.relu(z2) + v2) * (a2 - common.relu(z2) + v2))
    return obj


mnist = mnist()
# initialization
x_train = mnist.train.xs.astype(np.float32)
y_train = mnist.train.ys.astype(np.float32)
x_train = tf.transpose(x_train)
y_train = tf.transpose(y_train)
x_test = mnist.test.xs.astype(np.float32)
y_test = mnist.test.ys.astype(np.float32)
x_test = tf.transpose(x_test)
y_test = tf.transpose(y_test)

num_of_neurons = 1000
ITER = 200
index = 0
W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = Net(x_train, y_train, num_of_neurons)
u = tf.zeros(z3.shape)
t = 0
train_acc = np.zeros(ITER)
test_acc = np.zeros(ITER)
linear_r = np.zeros(ITER)
objective_value = np.zeros(ITER)
train_cost = np.zeros(ITER)
test_cost = np.zeros(ITER)
rho = 1e-06
tau = 1
theta = 1
t = 0
flag = 0
count = 0
# dlADMM
for i in range(ITER):
    pre = time.time()
    print("iter=", i)
    z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
    b3 = common.update_b(a2, W3, z3, b3, u, rho)
    W3 = common.update_W(a2, b3, z3, W3, u, rho, theta)
    a2 = common.update_a(W3, b3, z3, z2, a2, u, 0, rho, tau)
    z2 = common.update_z(a1, W2, b2, a2, z2, 0, 0, rho)
    b2 = common.update_b(a1, W2, z2, b2, 0, rho)
    W2 = common.update_W(a1, b2, z2, W2, 0, rho, theta)
    a1 = common.update_a(W2, b2, z2, z1, a1, 0, 0, rho, tau)
    z1 = common.update_z(x_train, W1, b1, a1, z1, 0, 0, rho)
    b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
    W1 = common.update_W(x_train, b1, z1, W1, 0, rho, theta)
    W1 = common.update_W(x_train, b1, z1, W1, 0, rho, theta)
    b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
    z1 = common.update_z(x_train, W1, b1, a1, z1, 0, 0, rho)
    a1 = common.update_a(W2, b2, z2, z1, a1, 0, 0, rho, tau)
    W2 = common.update_W(a1, b2, z2, W2, 0, rho, theta)
    b2 = common.update_b(a1, W2, z2, b2, 0, rho)
    z2 = common.update_z(a1, W2, b2, a2, z2, 0, 0, rho)
    a2 = common.update_a(W3, b3, z3, z2, a2, u, 0, rho, tau)
    W3 = common.update_W(a2, b3, z3, W3, u, rho, theta)
    b3 = common.update_b(a2, W3, z3, b3, u, rho)
    z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
    u = u + rho * (z3 - tf.matmul(W3, a2) - b3)
    print("Time per iteration:", time.time() - pre)
    r1 = tf.reduce_sum((z1 - tf.matmul(W1, x_train) - b1) * (z1 - tf.matmul(W1, x_train) - b1))
    r2 = tf.reduce_sum((z2 - tf.matmul(W2, a1) - b2) * (z2 - tf.matmul(W2, a1) - b2))
    r3 = tf.reduce_sum((z3 - tf.matmul(W3, a2) - b3) * (z3 - tf.matmul(W3, a2) - b3))
    linear_r[i] = r3

    obj = objective(x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, 0, 0, rho)
    print("obj=", obj.numpy())
    objective_value[i] = obj.numpy()
    print("r1=", r1.numpy())
    print("r2=", r2.numpy())
    print("r3=", r3.numpy())
    print("rho=", rho)
    (train_acc[i], train_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, x_train, y_train)
    print("training cost:", train_cost[i])
    print("training acc:", train_acc[i])
    (test_acc[i], test_cost[i]) = test_accuracy(W1, b1, W2, b2, W3, b3, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:", test_acc[i])
    if i > 2 and train_cost[i] > train_cost[i - 1] and train_cost[i - 1] > train_cost[i - 2] and train_cost[i - 2] > \
            train_cost[i - 3]:
        rho = np.minimum(10 * rho, 0.1)
        if num_of_neurons >= 100:
            tau = tau * 10
            theta = theta * 10
