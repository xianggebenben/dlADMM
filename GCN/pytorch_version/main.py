import torch
import numpy as np
import argparse
import time
import GCN.pytorch_version.dlADMM.common as common
import GCN.pytorch_version.dlADMM.input_data as input_data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mem_init = torch.cuda.memory_allocated(device=device)
# =================== parameters ===================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset: cora, pubmed, citeseer, coauthor_cs, coauthor_physics.')
parser.add_argument('--rho', type=float, default=1e-3,
                    help='rho')
args = parser.parse_args()


# =================== fixed random seed ===================
torch.manual_seed(args.seed)

# =================== import dataset ===================
if args.dataset == 'cora':
    data = input_data.cora()
    mu = 0
elif args.dataset == 'pubmed':
    data = input_data.pubmed()
    mu = 0
elif args.dataset == 'citeseer':
    data = input_data.citeseer()
    mu = 100
elif args.dataset == 'coauthor_cs':
    data = input_data.coauthor_cs()
    mu = 10
elif args.dataset == 'coauthor_physics':
    data = input_data.coauthor_physics()
    mu = 10
print("dataset=",args.dataset)
print("neurons=",args.hidden)
print("rho=",args.rho)
rho=args.rho
data.x = data.x.to(device)
data.adj = data.adj.to(device)
data.label_train = data.label_train.to(device)
data.label_test = data.label_test.to(device)
data.label_train_onehot = data.label_train_onehot.to(device)
adj_train = common.sparse_mask(data.adj, data.train_mask).to(device)
w1, z1, w2, z2 = common.gcn(
    adj=data.adj, x=data.x, nfeat=data.num_features, nhid=args.hidden, nclass=data.num_classes, seed=args.seed)

w1 = w1.to(device)
z1 = z1.to(device)
w2 = w2.to(device)
z2 = z2.to(device)
y2 = torch.zeros(z2.shape, device=device)

z1[data.test_mask] = 0
z2[data.test_mask] = 0

admm_train_loss = np.zeros(args.epochs)
admm_train_acc = np.zeros(args.epochs)
admm_test_loss = np.zeros(args.epochs)
admm_test_acc = np.zeros(args.epochs)
pres = np.zeros(args.epochs)
dres = np.zeros(args.epochs)
obj = np.zeros(args.epochs)

time_avg = 0
for epoch in range(args.epochs):
    pre = time.time()
    print("----------------------------------------------")
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    print("iter={:3d}".format(epoch))
    # ================ backward update =====================
    z2 = common.update_z2(adj_train, z1, z2, w2, y2, rho, data.label_train_onehot, data.train_mask)  # modified!
    w2 = common.update_w2(adj_train, z1, z2, w2, y2, rho, data.train_mask, mu)
    z1 = common.update_z1(adj_train, data.x, z1, z2, w1, w2, y2, rho, data.train_mask)
    w1 = common.update_w1(adj_train, data.x, z1, w1, rho, data.train_mask, mu)
    #
    # # ================ forward update =====================
    w1 = common.update_w1(adj_train, data.x, z1, w1, rho, data.train_mask, mu)
    z1 = common.update_z1(adj_train, data.x, z1, z2, w1, w2, y2, rho, data.train_mask)
    w2 = common.update_w2(adj_train, z1, z2, w2, y2, rho, data.train_mask, mu)
    z2_old = z2
    z2 = common.update_z2(adj_train, z1, z2, w2, y2, rho, data.label_train_onehot, data.train_mask)
    pre_res = time.time()
    r = z2[data.train_mask] - common.azw(adj_train, z1, w2)
    pres[epoch] = torch.norm(r ** 2)

    # dual residual
    d = rho * (z2 - z2_old)[data.train_mask]
    dres[epoch] = torch.norm(d ** 2)
    res_time = time.time() - pre_res
    y2 = common.update_y2(data.adj, z1, z2, w2, y2, rho)
    time_iter = time.time() - pre - res_time
    time_avg += time_iter / args.epochs
    admm_train_acc[epoch], admm_test_acc[epoch], admm_train_loss[epoch], admm_test_loss[epoch] \
        = common.test(data.adj, data.x, w1, w2, data.label_train, data.label_test, data.train_mask, data.test_mask)
    obj[epoch] = common.objective(adj_train, data.x, z1, z2, w1, w2, y2, rho,mu, data.label_train_onehot, mask=data.train_mask)
    print("obj={}".format(obj[epoch]))
    if (epoch + 1) % 50 == 0:
         rho = np.minimum(10 * rho, 1)
    print("Time per iteration:", time_iter)
    print("rho=", rho)
    print("Train loss: {:.6f}. Train acc: {:.6f}."
          "Test loss: {:.6f}. Test acc: {:.6f}.".format(
        admm_train_loss[epoch], admm_train_acc[epoch], admm_test_loss[epoch], admm_test_acc[epoch]))
