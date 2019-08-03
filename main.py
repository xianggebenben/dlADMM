import cupy as np
from cupy import matmul as mul
import sys
import time
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from dlADMM import common
from dlADMM.input_data import mnist,fashion_mnist


# initialize the neural network
def Net(images, label, num_of_neurons):
    seed_num = 13
    np.random.seed(seed=seed_num)
    W1 = np.random.normal(0, 0.1, size=(num_of_neurons, 28 * 28))
    np.random.seed(seed=seed_num)
    b1 = np.random.normal(0, 0.1, size=(num_of_neurons, 1))
    z1 = np.matmul(W1, images) + b1
    a1 = common.relu(z1)
    np.random.seed(seed=seed_num)
    W2 = np.random.normal(0, 0.1, size=(num_of_neurons, num_of_neurons))
    np.random.seed(seed=seed_num)
    b2 = np.random.normal(0, 0.1, size=(num_of_neurons, 1))
    z2 = np.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    np.random.seed(seed=seed_num)
    W3 = np.random.normal(0, 0.1, size=(10, num_of_neurons))
    np.random.seed(seed=seed_num)
    b3 = np.random.normal(0, 0.1, size=(10, 1))
    z3 = np.ones(label.shape)
    z3[label == 0] = -1
    z3[label == 1] = 1
    return W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3


# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3, images, labels):
    nums = labels.shape[1]
    z1 = np.matmul(W1, images) + b1
    a1 = common.relu(z1)
    z2 = np.matmul(W2, a1) + b2
    a2 = common.relu(z2)
    z3 = np.matmul(W3, a2) + b3
    cost = common.cross_entropy_with_softmax(labels, z3) / nums
    pred = np.argmax(labels, axis=0)
    label = np.argmax(z3, axis=0)
    return (np.sum(np.equal(pred, label)) / nums, cost)


# return the value of the augmented Lagrangian
def objective(x_train, y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u, v1, v2, rho):
    r1 = np.sum((z1 - mul(W1, x_train) - b1) * (z1 - mul(W1, x_train) - b1))
    r2 = np.sum((z2 - mul(W2, a1) - b2) * (z2 - mul(W2, a1) - b2))
    r3 = np.sum((z3 - mul(W3, a2) - b3) * (z3 - mul(W3, a2) - b3))
    loss = common.cross_entropy_with_softmax(y_train, z3)
    obj = loss + np.trace(mul(z3 - mul(W3, a2) - b3, np.transpose(u)))
    obj = obj + rho / 2 * r1 + rho / 2 * r2 + rho / 2 * r3
    obj = obj + rho / 2 * np.sum((a1 - common.relu(z1) + v1) * (a1 - common.relu(z1) + v1)) + rho / 2 * np.sum(
        (a2 - common.relu(z2) + v2) * (a2 - common.relu(z2) + v2))
    return obj
mnist = mnist()
#initialization
x_train = mnist.train.xs
y_train = mnist.train.ys
x_train = np.swapaxes(x_train, 0, 1)
y_train = np.swapaxes(y_train, 0, 1)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = mnist.test.xs
y_test = mnist.test.ys
x_test = np.swapaxes(x_test, 0, 1)
y_test = np.swapaxes(y_test, 0, 1)
x_test = np.array(x_test)
y_test = np.array(y_test)

num_of_neurons = 1000
ITER = 10000
index = 0
W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3 = Net(x_train, y_train, num_of_neurons)
u=np.zeros(z3.shape)
t=0
train_acc=np.zeros(ITER)
test_acc=np.zeros(ITER)
linear_r=np.zeros(ITER)
objective_value=np.zeros(ITER)
train_cost=np.zeros(ITER)
test_cost=np.zeros(ITER)
rho =1e-06
tau1 =1
tau2 =1
theta1 =1
theta2 =1
theta3 =1
t=0
flag =0
count =0
# dlADMM
for i in range(ITER):
    pre = time.time()
    print("iter=",i)
    z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
    b3 = common.update_b(a2, W3, z3, b3, u, rho)
    W3 = common.update_W(a2, b3, z3, W3, u, rho,theta3)
    a2 = common.update_a(W3, b3, z3, z2, a2, u, 0,rho,tau2)
    z2 = common.update_z(a1, W2, b2, a2, z2, 0, 0,rho)
    b2 = common.update_b(a1, W2, z2, b2, 0, rho)
    W2 = common.update_W(a1, b2, z2, W2, 0, rho,theta2)
    a1 = common.update_a(W2, b2, z2, z1, a1, 0, 0,rho,tau1)
    z1 = common.update_z(x_train, W1, b1, a1, z1, 0, 0,rho)
    b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
    W1 = common.update_W(x_train, b1, z1, W1, 0,rho,theta1)
    W1 = common.update_W(x_train, b1, z1, W1, 0,rho,theta1)
    b1 = common.update_b(x_train, W1, z1, b1, 0, rho)
    z1 = common.update_z(x_train, W1, b1, a1,z1,0,0,rho)
    a1 = common.update_a(W2, b2, z2, z1, a1, 0,0,rho,tau1)
    W2 = common.update_W(a1, b2, z2, W2, 0,rho,theta2)
    b2 = common.update_b(a1, W2, z2, b2, 0,rho)
    z2 = common.update_z(a1, W2, b2, a2,z2,0,0,rho)
    a2 = common.update_a(W3, b3, z3, z2, a2, u,0,rho,tau2)
    W3 = common.update_W(a2, b3, z3, W3, u,rho,theta3)
    b3 = common.update_b(a2, W3, z3, b3, u,rho)
    z3 = common.update_zl(a2, W3, b3, y_train, z3, u, rho)
    u = u +rho*(z3-mul(W3,a2)-b3)
    r1 =np.sum((z1 - mul(W1, x_train) - b1) * (z1 - mul(W1, x_train) - b1))
    r2 =np.sum((z2 - mul(W2, a1) - b2) * (z2 - mul(W2, a1) - b2))
    r3 = np.sum((z3 - mul(W3, a2) - b3) * (z3 - mul(W3, a2) - b3))
    linear_r[i] = r3


    obj=objective(x_train,y_train, W1, b1, z1, a1, W2, b2, z2, a2, W3, b3, z3, u,0,0,rho)
    print("obj=",obj)
    objective_value[i]=obj
    print("r1=",r1)
    print("r2=",r2)
    print("r3=",r3)
    print("rho=",rho)
    (train_acc[i],train_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3, x_train,y_train)
    print("training cost:", train_cost[i])
    print("training acc:",train_acc[i])
    (test_acc[i],test_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:",test_acc[i])
    print("Time per iteration:", time.time() - pre)
    if i>2 and train_cost[i]>train_cost[i-1] and train_cost[i-1]>train_cost[i-2] and train_cost[i-2]>train_cost[i-3]:
        rho=np.minimum(10*rho,0.1)
        if num_of_neurons>=100:
            tau1 = tau1 * 10
            tau2 = tau2 * 10
            theta1 = theta1 * 10
            theta2 = theta2 * 10
            theta3 = theta3 * 10
