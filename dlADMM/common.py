import cupy as np
from cupy import matmul as mul
# return softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
# return cross entropy
def cross_entropy(label, prob):
    loss = -np.sum(label * np.log(prob))
    return loss
# return the cross entropy loss function
def cross_entropy_with_softmax(label, z):
    prob = softmax(z)
    loss = cross_entropy(label, prob)
    return loss
#return the  relu function
def relu(x):
    return np.maximum(x, 0)
# return phi
def eq1(a, W_next, b_next, z_next, u_next,rho):
    temp = z_next - mul(W_next, a) - b_next+u_next/rho
    res = rho / 2 * np.sum(temp * temp)
    return res
# return the derivative of phi with regard to a
def eq1_a(a, W_next, b_next, z_next, u_next,rho):
    res = rho * mul(np.transpose(W_next), mul(W_next, a) + b_next - z_next-u_next/rho)
    return res
# return the derivative of phi with regard to W
def eq1_W(a, W_next, b_next, z_next, u_next,rho):
    temp = mul(W_next, a) + b_next - z_next-u_next/rho
    temp2 = a.T
    res = rho * mul(temp, temp2)
    return res
# return the derivative of phi with regard to b
def eq1_b(a, W_next, b_next, z_next,u_next, rho):
    res = np.mean(rho * (mul(W_next, a) + b_next - z_next-u_next/rho), axis=1).reshape(-1, 1)
    return res
# return the derivative of phi with regard to z
def eq1_z(a, W_next, b_next, z_next, u_next, rho):
    res = rho * (z_next - b_next - mul(W_next, a)+u_next/rho)
    return res
# return the quadratic approximation of W-subproblem
def P(W_new, theta, a_last, W, b, z, u,rho):
    temp = W_new - W
    res = eq1(a_last, W, b, z, u,rho) + np.sum(eq1_W(a_last, W, b, z, u,rho) * temp) + np.sum(theta * temp * temp) / 2
    return res
# return the quadratic approximation of a-subproblem
def Q(a_new, tau, a, W_next, b_next, z_next, u_next,v,z,rho):
    temp = a_new - a
    res = a_obj(a, W_next, b_next, z_next, u_next,v,z,rho) + np.sum(a_obj_gradient(a, W_next, b_next, z_next, u_next,v,z,rho) * temp) + np.sum(
        tau * temp * temp) / 2
    return res
# return the objective value of a-subproblem
def a_obj(a, W_next, b_next, z_next, u_next,v,z,rho):
    res=eq1(a, W_next, b_next, z_next, u_next,rho)+rho/2*np.sum((a-relu(z)+v)*(a-relu(z)+v))
    return res
# return the gradient of a-subproblem
def a_obj_gradient(a, W_next, b_next, z_next, u_next,v,z,rho):
    res=eq1_a(a, W_next, b_next, z_next, u_next,rho)+rho*(a-relu(z)+v)
    return res
# return the result of W-subproblem
def update_W(a_last, b, z, W_old, u,rho):
    gradients = eq1_W(a_last, W_old, b, z, u,rho)
    gamma = 2
    alpha = 1
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
    f=(z-mul(W,a_last)-b+u/rho)*(z-mul(W,a_last)-b+u/rho)+(a-relu(z)+v)*(a-relu(z)+v)
# return the result of W-subproblem
def update_W(a_last, b, z, W_old, u,rho):
    gradients = eq1_W(a_last, W_old, b, z, u,rho)
    gamma = 2
    alpha = 1
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
    f=(z-mul(W,a_last)-b+u/rho)*(z-mul(W,a_last)-b+u/rho)+(a-relu(z)+v)*(a-relu(z)+v)
    return f
# return the result of z-subproblem
def update_z(a_last, W, b, a,z_old, u,v,rho):
    z1=np.matmul(W,a_last)+b-u/rho;
    z2=(z1+a+v)/2
    z1=np.minimum(z1,0)
    z2=np.maximum(z2,0)
    value1=z_obj(a_last, W, b, z1, u,v,a,rho)
    value2=z_obj(a_last, W, b, z2, u,v,a,rho)
    z=z1
    z[value1>value2]=z2[value1>value2]
    return z
# return the result of z_L-subproblem by FISTA
def update_zl(a_last, W, b, label, zl_old, u,rho):
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 10e-5
    for i in range(MAX_ITER):
        fzl_old = fzl
        fzl = cross_entropy_with_softmax(label, zl)+rho/2*np.sum((zl-mul(W,a_last)-b+u/rho)*(zl-mul(W,a_last)-b+u/rho))
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (mul(W, a_last)+b-u/rho) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl
# return the result of a-subproblem
def update_a(W_next, b_next, z_next, z, a_old, u_next,v,rho):
    gradient = a_obj_gradient(a_old, W_next, b_next, z_next, u_next, v,z,rho)
    eta = 2
    t=1
    beta=a_old-gradient/t
    while (a_obj(beta, W_next, b_next, z_next, u_next,v,z,rho) > Q(beta, t, a_old, W_next, b_next, z_next, u_next,v,z,rho)):
        t = t * eta
        beta =a_old-gradient/t
    tau = t
    a = beta
    return a