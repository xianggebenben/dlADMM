import torch.nn.functional as F
import torch
from torch_sparse import SparseTensor, spmm
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_sparse import SparseTensor, fill_diag, sum, mul, cat, masked_select
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch.autograd import Variable
from torch_geometric.utils.num_nodes import maybe_num_nodes
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# normalize adj matrix
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        # return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        n = torch.max(edge_index[0])+1  # num of nodes
        adj_norm = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
                                  sparse_sizes=(n, n))
        return adj_norm

# initialize network parameters
def gcn(adj, x, nfeat, nhid, nclass,seed):
    torch.manual_seed(seed)
    w1 = Parameter(torch.Tensor(nfeat, nhid)).to(device)
    glorot(w1)
    torch.manual_seed(seed)
    w2 = Parameter(torch.Tensor(nhid, nclass)).to(device)
    glorot(w2)
    z1 = azw(adj, x, w1)
    z1 = F.relu(z1)
    z2 = azw(adj, z1, w2)
    return w1, z1, w2, z2



def azw(adj, z, w):
    if isinstance(adj, SparseTensor):
        row, col, adj_value = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        row_size = adj.size(dim=0)
        col_size = adj.size(dim=1)
        f1, f2 = w.size()
        if f1 - f2 > 0:  # A(ZW) would be more efficient
            temp = z.matmul(w)
            return spmm(edge_index, adj_value, row_size, col_size, temp)
        else:  # (AZ)W would be more efficient
            temp = spmm(edge_index, adj_value, row_size, col_size, z)
            return temp.matmul(w)
    elif adj == 0:
        return 0
    else:  # adj is dense matrix
        return adj.matmul(z.matmul(w))


def forwardprop(adj, x, w1, w2):
    z1 = F.relu(azw(adj, x, w1))
    z2 = azw(adj, z1, w2)
    return z2


# =================== test accuracy ===================
def test(adj, x, w1, w2, label_train, label_test, mask_train, mask_test):
    pred = forwardprop(adj, x, w1, w2)
    pred_train = pred[mask_train]
    pred_test = pred[mask_test]
    pred_train = F.log_softmax(pred_train, dim=1)
    pred_test = F.log_softmax(pred_test, dim=1)
    loss_train = F.nll_loss(pred_train, label_train)
    loss_test = F.nll_loss(pred_test, label_test)

    pred_train = pred_train.argmax(dim=1)
    pred_test = pred_test.argmax(dim=1)
    acc_train = pred_train.eq(label_train)
    acc_test = pred_test.eq(label_test)
    acc_train = float(acc_train.sum().item())
    acc_test = float(acc_test.sum().item())
    length_train = mask_train.sum().item()
    length_test = mask_test.sum().item()
    acc_train = acc_train / length_train
    acc_test = acc_test / length_test
    return acc_train, acc_test, loss_train, loss_test


# augmented lagrangian
def objective(adj_train, x, z1, z2, w1, w2, y2, rho, mu,label_onehot, mask):  # modified!
    temp1 = z1[mask] - F.relu(azw(adj_train, x, w1))
    temp2 = z2[mask] - azw(adj_train, z1, w2)
    loss, _ = cross_entropy_with_softmax(z2[mask], label_onehot)  # modified!
    # z2 = F.log_softmax(z2[mask], dim=1)
    # loss = F.nll_loss(z2, label, reduction="sum")
    obj = loss + rho / 2 * torch.norm(temp1 ** 2) + rho / 2 * torch.norm(temp2 ** 2) \
          + torch.einsum("ij, ij ->", y2[mask], temp2)+mu*torch.norm(w1)**2+mu*torch.norm(w2)**2
    return obj


def sparse_mask(adj, train_mask):
    return masked_select(adj, 0, train_mask)


def phi_z1(adj_train, z1, z2, w2, y2, rho, temp1,train_mask):
    temp1 = z1[train_mask] - temp1
    temp2 = (z2[train_mask]-azw(adj_train, z1, w2))
    return torch.einsum("ij, ij ->", (rho/2 * temp1), temp1) \
           + torch.einsum("ij, ij ->", (y2[train_mask]+rho/2 * temp2), temp2)


def u_z1(z1_old, z1_new, tau, gradient,train_mask, loss):
    res = z1_new[train_mask]-z1_old[train_mask]
    return loss + torch.einsum("ij, ij ->", (gradient + tau / 2 * res), res)


def update_z1(adj_train, x, z1_old, z2, w1, w2, y2, rho,train_mask):
    temp1 = F.relu(azw(adj_train, x, w1))
    loss = phi_z1(adj_train, z1_old, z2, w2, y2, rho, temp1,train_mask)
    gradient = torch.autograd.grad(loss, z1_old)[0][train_mask]
    eta = 2
    t = 0.001
    count = 0
    beta =z1_old.clone()
    beta[train_mask] = z1_old[train_mask] - gradient/t
    while (phi_z1(adj_train, beta, z2, w2, y2, rho, temp1,train_mask)
            > u_z1(z1_old, beta, t,gradient,train_mask, loss)):
        t = t * eta
        beta[train_mask] = z1_old[train_mask]-gradient/t
        count += 1
        if count > 32:
            beta[train_mask] = z1_old[train_mask]
            break
    return beta



def cross_entropy_with_softmax(zl, label_onehot):
    # prob = F.log_softmax(zl, dim=1)
    # loss = F.nll_loss(prob, label, reduction="sum")
    prob = F.softmax(zl, dim=1)
    loss = - torch.einsum("ij, ij->", label_onehot, torch.log(prob+1e-10))
    return loss, prob


def phi_z2(z2,  y2, rho, temp):
    temp = z2 - temp
    return rho/2 * torch.norm((temp + y2/rho) ** 2)


# cross-entropy loss
def update_z2(adj_train, z1, z2_old, w2, y2, rho, label_train_one_hot, train_mask):
    fz2 = 10e10
    MAX_ITER = 50
    z2 = z2_old
    lamda = 1
    zeta = z2[train_mask]
    eta = 5
    TOLERANCE = 1e-3
    temp = azw(adj_train, z1, w2)
    z2=z2.detach()
    for i in range(MAX_ITER):
        fz2_old = fz2
        loss, prob = cross_entropy_with_softmax(z2[train_mask,:], label_train_one_hot)
        fz2 = loss + phi_z2(z2[train_mask], y2[train_mask], rho, temp)
        if abs(fz2 - fz2_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradient2 = prob - label_train_one_hot
        zeta_old = zeta
        zeta = (-y2[train_mask] + rho*temp + z2[train_mask]/eta - gradient2)/(rho + 1 / eta)
        z2[train_mask,:] = (1 - gamma) * zeta + gamma * zeta_old
    return z2


def phi_wl(adj_train, zl, zl_1, wl, rho,train_mask, mu):
    temp = zl[train_mask]-F.relu(azw(adj_train, zl_1, wl))
    if mu == 0:
        return torch.einsum("ij, ij ->", (rho/2 * temp), temp)
    else:
        return torch.einsum("ij, ij ->", (rho / 2 * temp), temp) + mu / 2 * torch.norm(wl ** 2)


def u_wl(wl_old, wl_new, tau, gradient, loss):
    temp = wl_new - wl_old
    f = loss + torch.einsum("ij, ij ->", (gradient+ tau / 2 * temp), temp)
    return f


def update_w1(adj_train, x, z1, w1_old, rho,train_mask, mu):
    loss = phi_wl(adj_train, z1, x, w1_old, rho,train_mask, mu)
    gradient = torch.autograd.grad(loss, w1_old)[0]
    eta = 2
    t = 1
    count = 0
    beta = w1_old - gradient/t
    while phi_wl(adj_train, z1, x, beta, rho, train_mask, mu) \
            > u_wl(w1_old, beta, t, gradient, loss):
        t = t*eta
        beta = w1_old -gradient/t
        count += 1
        if count > 32:
            beta = w1_old
            break
    return beta


def phi_w2(adj_train, zl, zl_1, wl, yl, rho,train_mask, mu):
    temp = (zl[train_mask]-azw(adj_train, zl_1, wl))
    if mu == 0:
        return torch.einsum("ij, ij ->", (yl[train_mask] + rho / 2 * temp), temp)

    else:
        return torch.einsum("ij, ij ->", (yl[train_mask] + rho / 2 * temp), temp) + mu / 2 * torch.norm(wl ** 2)


def u_w2(wL_old, wL_new, tau, gradient, loss):
    temp = wL_new - wL_old
    if tau == 0:
        return loss + torch.einsum("ij, ij ->", gradient, temp)
    else:
        return loss + torch.einsum("ij, ij ->", gradient + tau / 2 * temp, temp)


def update_w2(adj_train, z1, z2, w2_old, y2, rho,train_mask, mu):
    w2_old = Variable(w2_old, requires_grad=True)
    loss = phi_w2(adj_train,z2, z1, w2_old, y2, rho,train_mask, mu)
    gradient = torch.autograd.grad(loss, w2_old)[0]
    eta = 2
    t = 0.01
    count = 0
    beta = w2_old - gradient/t
    while phi_w2(adj_train,z2, z1, beta, y2, rho,train_mask, mu) \
            > u_w2(w2_old, beta, t, gradient, loss):
        t = t*eta
        beta = w2_old -gradient/t
        count += 1
        if count > 32:
            beta = w2_old
            break
    return beta


def update_y2(adj, z1, z2, w2, y2, rho):
    temp1 = azw(adj, z1, w2)
    y2 = y2 + rho * (z2 - temp1)
    return y2
#
# def update_y2(adj_train, z1, z2, w2, y2, rho, train_mask):
#     temp1 = azw(adj_train, z1, w2)
#     y2[train_mask] = y2[train_mask] + rho * (z2[train_mask] - temp1)
#     return y2


