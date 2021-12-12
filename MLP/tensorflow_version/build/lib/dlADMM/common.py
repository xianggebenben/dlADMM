import tensorflow as tf
import numpy as np
# return softmax
def cross_entropy_with_softmax(label, zl):
    prob = softmax(zl)
    imask =tf.equal(prob,0.0)
    prob = tf.where(imask,1e-10,prob)
    loss = cross_entropy(label, prob)
    return loss
def softmax(x):
    exp =tf.math.exp(x)
    imask =tf.equal(exp,float("inf"))
    exp = tf.where(imask,tf.math.exp(88.6),exp)
    return exp/(tf.math.reduce_sum(exp,axis=0)+1e-10)
def cross_entropy(label, prob):
    loss = -tf.math.reduce_sum(label * tf.math.log(prob))
    return loss
#return the  relu function
def relu(x):
    return tf.maximum(x, 0)
# return phi
def eq1(a, W_next, b_next, z_next, u_next,rho):
    temp = z_next - tf.matmul(W_next, a) - b_next+u_next/rho
    res = rho / 2 * tf.reduce_sum(temp * temp)
    return res
# return the derivative of phi with regard to a
def eq1_a(a, W_next, b_next, z_next, u_next,rho):
    res = rho * tf.matmul(tf.transpose(W_next), tf.matmul(W_next, a) + b_next - z_next-u_next/rho)
    return res
# return the derivative of phi with regard to W
def eq1_W(a, W_next, b_next, z_next, u_next,rho):
    temp = tf.matmul(W_next, a) + b_next - z_next-u_next/rho
    temp2 = tf.transpose(a)
    res = rho * tf.matmul(temp, temp2)
    return res
# return the derivative of phi with regard to b
def eq1_b(a, W_next, b_next, z_next,u_next, rho):
    res = tf.reshape(tf.math.reduce_mean(rho * (tf.matmul(W_next, a) + b_next - z_next-u_next/rho), axis=1),shape=(-1, 1))
    return res
# return the derivative of phi with regard to z
def eq1_z(a, W_next, b_next, z_next, u_next, rho):
    res = rho * (z_next - b_next - tf.matmul(W_next, a)+u_next/rho)
    return res
# return the quadratic approximation of W-subproblem
def P(W_new, theta, a_last, W, b, z, u,rho):
    temp = W_new - W
    res = eq1(a_last, W, b, z, u,rho) + tf.reduce_sum(eq1_W(a_last, W, b, z, u,rho) * temp) + tf.reduce_sum(theta * temp * temp) / 2
    return res
# return the quadratic approximation of a-subproblem
def Q(a_new, tau, a, W_next, b_next, z_next, u_next,v,z,rho):
    temp = a_new - a
    res = a_obj(a, W_next, b_next, z_next, u_next,v,z,rho) + tf.reduce_sum(a_obj_gradient(a, W_next, b_next, z_next, u_next,v,z,rho) * temp) + tf.reduce_sum(
        tau * temp * temp) / 2
    return res
# return the objective value of a-subproblem
def a_obj(a, W_next, b_next, z_next, u_next,v,z,rho):
    res=eq1(a, W_next, b_next, z_next, u_next,rho)+rho/2*tf.reduce_sum((a-relu(z)+v)*(a-relu(z)+v))
    return res
# return the gradient of a-subproblem
def a_obj_gradient(a, W_next, b_next, z_next, u_next,v,z,rho):
    res=eq1_a(a, W_next, b_next, z_next, u_next,rho)+rho*(a-relu(z)+v)
    return res
# return the result of W-subproblem
def update_W(a_last, b, z, W_old, u,rho,alpha=1):
    gradients = eq1_W(a_last, W_old, b, z, u,rho)
    gamma = 2
    zeta = W_old - gradients / alpha
    while (eq1(a_last, zeta, b, z, u,rho) > P(zeta, alpha, a_last, W_old, b, z, u,rho)):
        alpha = alpha * gamma
        zeta = W_old - gradients / alpha  # Learning rate decreases to 0, leading to infinity loop here.
    theta = alpha
    W = zeta
    return W
# return the result of b-subproblem
def update_b(a_last, W, z, b_old, u,rho):
    gradients = eq1_b(a_last, W, b_old, z, u,rho)
    res = b_old - gradients / rho
    return res
# return the objective value of z-subproblem
def z_obj(a_last, W, b, z, u,v,a,rho):
    f=(z-tf.matmul(W,a_last)-b+u/rho)*(z-tf.matmul(W,a_last)-b+u/rho)+(a-relu(z)+v)*(a-relu(z)+v)
    return f
# return the result of z-subproblem
def update_z(a_last, W, b, a, u,v,rho):
    z1=tf.matmul(W,a_last)+b-u/rho;
    z2=(z1+a+v)/2
    z1=tf.minimum(z1,0)
    z2=tf.maximum(z2,0)
    value1=z_obj(a_last, W, b, z1, u,v,a,rho)
    value2=z_obj(a_last, W, b, z2, u,v,a,rho)
    imask =tf.greater(value1,value2)
    z=tf.where(imask,z2,z1)
    return z
# return the result of z_L-subproblem by FISTA
def update_zl(a_last, W, b, label, zl_old, u,rho):
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        fzl_old = fzl
        fzl = cross_entropy_with_softmax(label, zl)+rho/2*tf.reduce_sum((zl-tf.matmul(W,a_last)-b+u/rho)*(zl-tf.matmul(W,a_last)-b+u/rho))
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (tf.matmul(W, a_last)+b-u/rho) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl
# return the result of a-subproblem
def update_a(W_next, b_next, z_next, z, a_old, u_next,v,rho,t=1):
    gradient = a_obj_gradient(a_old, W_next, b_next, z_next, u_next, v,z,rho)
    eta = 2
    beta=a_old-gradient/t
    while (a_obj(beta, W_next, b_next, z_next, u_next,v,z,rho) > Q(beta, t, a_old, W_next, b_next, z_next, u_next,v,z,rho)):
        t = t * eta
        beta =a_old-gradient/t
    tau = t
    a = beta
    return a