from email.policy import default
from turtle import forward
# from matplotlib.font_manager import _Weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.nn import GATConv, GraphConv, APPNPConv, SGConv, ChebConv, SAGEConv
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.utils import JumpingKnowledge
import numpy as np
import argparse
from dataset_utils import preprocess_data
import os.path as osp
import os
import random
from utils import random_planetoid_splits
from tqdm import tqdm


class GPR_prop_attention(nn.Module):
    def __init__(self, K, alpha, Init, hidden_dim, attention_dim, Gamma=None):
        super(GPR_prop_attention, self).__init__()
        self.K = K
        self.Init = Init
        self.alpha = alpha
        

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        
        self.temp = nn.Parameter(torch.tensor(TEMP))
        # self.fc = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1, bias=False)


    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
    
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        e = self.attn_fc(z2).squeeze()
        return {'e': F.leaky_relu(e)}
    
    # def edge_norm(self, edges):
        # return {'e': edges.data['e'] * edges.src['d'] * edges.dst['d']}

    def forward(self, g, in_feat):
        with g.local_scope():
            g.ndata['h'] = in_feat
            g.ndata['TH'] = in_feat * (self.temp[0])
            for k in range(self.K):
                g.ndata['z'] = g.ndata['h']
                g.apply_edges(self.edge_attention)
                g.edata['e'] = edge_softmax(g, g.edata['e'])
                # g.apply_edges(self.edge_norm)
                g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
                gamma = self.temp[k+1]
                g.ndata['TH'] = g.ndata['TH'] + g.ndata['h'] * gamma
                # print(self.temp)
            return g.ndata['TH']


class GPR_prop(nn.Module):
    def __init__(self, K, alpha, Init, Dropout, Gamma=None):
        super(GPR_prop, self).__init__()
        self.K = K
        self.Init = Init
        self.alpha = alpha
        # self.norm = EdgeWeightNorm(norm='both')
        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma
        # self.g = g
        self.temp = nn.Parameter(torch.tensor(TEMP))
        self.dropout = nn.Dropout(Dropout)


    
    def reset_parameters(self):
        return 0
        # torch.nn.init.zeros_(self.temp)
        # for k in range(self.K+1):
        #     self.temp.data[k] = self.alpha*(1-self.alpha)**k
        # self.temp.data[-1] = (1-self.alpha)**self.K

    def edge_applying(self, edges):
        norm = edges.dst['d'] * edges.src['d']
        # norm = self.dropout(norm)
        return {'norm': norm}
        
    def forward(self, g, in_feat):
        with g.local_scope():
            g.ndata['h'] = in_feat
            g.ndata['TH'] = in_feat * (self.temp[0])
            for k in range(self.K):
                g.apply_edges(self.edge_applying)
                g.update_all(fn.u_mul_e('h', 'norm', '_'), fn.sum('_', 'h'))
                gamma = self.temp[k+1]
                g.ndata['TH'] = g.ndata['TH'] + g.ndata['h'] * gamma
                # print(self.temp)
            return g.ndata['TH']

class GPRGNN(nn.Module):
    def __init__(self, graph, args):
        super(GPRGNN, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.lin1 = nn.Linear(self.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, self.num_classes)

        if args.passing_type == "Attention":
            self.prop1 = GPR_prop_attention(args.K, args.alpha, args.Init, len(g.ndata['label'].unique()), args.attention_dim, args.Gamma)
        elif args.passing_type == "Origin":
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.dropout, args.Gamma)
        elif args.passing_type == "APPNP":
            self.prop1 = APPNPConv(args.K, args.alpha)


        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
    
    def reset_parameters(self):
        if args.passing_type != "APPNP":
            self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    
    def forward(self, g, feat):
        with g.local_scope():
            g = dgl.add_self_loop(g)
            h = F.dropout(feat, p=self.dropout, training=self.training)
            h = F.relu(self.lin1(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lin2(h)

            if self.dprate == 0.0:
                h = self.prop1(g, h)
                return F.log_softmax(h, dim=1)
            else:
                h = F.dropout(h, p=self.dprate, training=self.training)
                h = self.prop1(g, h)
                return F.log_softmax(h, dim=1)



class GAT_Net(nn.Module):
    def __init__(self, graph, args):
        super(GAT_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = GATConv(
            self.num_features,
            args.hidden,
            num_heads=args.heads,
            attn_drop=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            self.num_classes,
            num_heads=args.output_heads,
            attn_drop=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, feat):
        with g.local_scope():
            h = feat
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv1(g, h)
            h = h.view(h.size(0), -1)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(g, h)
            h = h.view(h.size(0), -1)
            return F.log_softmax(h, dim=1)
            
class GCN_Net(nn.Module):
    def __init__(self, graph, args):
        super(GCN_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = GraphConv(self.num_features, args.hidden)
        self.conv2 = GraphConv(args.hidden, self.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, g, feat):
        h = feat
        h = F.relu(self.conv1(g, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return F.log_softmax(h, dim=1)


class APPNP_Net(nn.Module):
    def __init__(self, graph, args):
        super(APPNP_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.lin1 = nn.Linear(self.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, self.num_classes)
        self.prop1 = APPNPConv(args.K, args.alpha, args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)
        h = feat
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        h = self.prop1(g, h)
        return F.log_softmax(h, dim=1)

class SGC_Net(nn.Module):
    def __init__(self, graph, args):
        super(SGC_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = SGConv(self.num_features, self.num_classes, k=3)
        self.dropout = args.dropout
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
    
    def forward(self, g, feat):
        h = feat
        h = F.relu(self.conv1(g, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return F.log_softmax(h, dim=1)




class MLP_Net(nn.Module):
    def __init__(self, graph, args):
        super(MLP_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.lin1 = nn.Linear(self.num_features, args.hidden)
        self.lin2 = nn.Linear(args.hidden, self.num_classes)
        self.dropout = args.dropout
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, g, feat):
        h = feat
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)
        return F.log_softmax(h, dim=1)

class Cheby_Net(nn.Module):
    def __init__(self, graph, args):
        super(Cheby_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = ChebConv(self.num_features, args.hidden, 2)
        self.conv2 = ChebConv(args.hidden, self.num_classes, 2)
        self.dropout = args.dropout
        self.lambda_max = dgl.laplacian_lambda_max(graph)
    
    def reset_parameters(self):
        # self.conv1.reset_parameters()
        # self.conv2.reset_parameters()
        pass
    
    def forward(self, g, feat):
        h = feat
        h = F.relu(self.conv1(g, h, lambda_max=self.lambda_max))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h, lambda_max=self.lambda_max)
        return F.log_softmax(h, dim=1)

class SAGE_Net(nn.Module):
    def __init__(self, graph, args):
        super(SAGE_Net, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = SAGEConv(self.num_features, args.hidden, 'mean')
        self.conv2 = SAGEConv(args.hidden, self.num_classes, 'mean')
        self.dropout = args.dropout
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, g, feat):
        h = feat
        h = F.relu(self.conv1(g, h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(g, h)
        return F.log_softmax(h, dim=1)


class GCN_JKNet(nn.Module):
    def __init__(self, graph, args):
        # in_channels = dataset.num_features
        # out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.num_classes = len(g.ndata['label'].unique())
        self.num_features = graph.ndata['feat'].shape[1]
        self.conv1 = GraphConv(self.num_features, 16)
        self.conv2 = GraphConv(16, 16)
        self.lin1 = torch.nn.Linear(16, self.num_classes)
        self.one_step = APPNPConv(k=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   in_feats=16,
                                   num_layers=4
                                   )

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.JK.reset_parameters()

    def forward(self, g, feat):
        g = dgl.add_self_loop(g)
        h = feat
        h1 = F.relu(self.conv1(g, h))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.conv2(g, h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)

        h = self.JK([h1, h2])
        h = self.one_step(g, h)
        h = self.lin1(h)
        return F.log_softmax(h, dim=1)



def RunExp(args, g, model, optimizer):
    

    train_rate = args.train_ratio
    val_rate = args.val_ratio
    percls_trn = int(round(train_rate*len(g.ndata['label'])/g.ndata['label'].unique().size(0)))
    val_lb = int(round(val_rate*len(g.ndata['label'])))
    
    g = random_planetoid_splits(g, g.ndata['label'].unique().size(0), percls_trn=percls_trn, val_lb=val_lb, Flag=0)
    g = dgl.add_self_loop(g)
    
    loss_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    best_test_acc_list = []
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['label']
    best_val_acc = 0
    best_test_acc = 0
    for e in range(args.epoch + 1):
        model.train()
        logits = model(g, g.ndata['feat'])
        loss = F.nll_loss(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits = model(g, g.ndata['feat'])
        val_loss = F.nll_loss(logits[val_mask], labels[val_mask])
        pred = logits.argmax(1)
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean() 
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean() 
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean() 

        loss_list.append(loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        best_test_acc_list.append(best_test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
        # if e % 5 == 0:
        #     print('{} {}: In epoch {}, loss:{:.3f}, trian_acc:{:.3f}, val_acc:{:.3f} (best_val_acc:{:.3f}), test_acc:{:.3f} (best_test_acc:{:.3f})'.format(args.dataset, args.model, e, loss, train_acc, val_acc, best_val_acc, test_acc, best_test_acc))
        # if e % 100 == 0:
        #     if args.model == 'GPRGNN':
        #         print(model.prop1.temp)
        
    if args.model == "GPRGNN":
        Gamma_0 = model.prop1.temp.detach().numpy()
    else:
        Gamma_0 = [1 ,2]
    
    return best_test_acc, best_val_acc, Gamma_0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--attention_dim', type=int, default=80)
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--passing_type', type=str, default='Origin')
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--model', type=str, default="APPNP")
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    device = 'cpu'

###################
# dataset select
###################
    # dataset = dgl.data.CoraGraphDataset()
    # dataset = dgl.data.CoraGraphDataset()
    # dataset = dgl.data.CoraGraphDataset()
    # dataset = dgl.data.CiteseerGraphDataset()
    # dataset = dgl.data.PubmedGraphDataset()
    # g = dataset[0]
    random.seed(2022)
    g = preprocess_data(args.dataset)
    g = g.to(device)
    if args.model in ['GPRGNN', 'GAT']:
        g = dgl.add_self_loop(g)
    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm
    # print(g.ndata['feat'].shape)
    # print(g.ndata['label'].shape)
    # p
    # 
    # rint(g.ndata['train_mask'].shape)
    # print(g.ndata['val_mask'])



    # g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.train_ratio)


###################
# model select
###################
    existing_model = ['GPRGNN', "APPNP",'GAT', 'GCN', 'MLP', "JKNet", "SGC", "GCN-Cheby", "SAGE"]
    if args.model not in existing_model:
            raise ValueError(
                f'name of model must be one of: {existing_model}')
    else:
        if args.model == "GPRGNN":
            model = GPRGNN(g, args).to(device)
        elif args.model == "APPNP":
            model = APPNP_Net(g, args).to(device)
        elif args.model == "GAT":
            model = GAT_Net(g, args).to(device)
        elif args.model == "GCN":
            model = GCN_Net(g, args).to(device)
        elif args.model == "MLP":
            model = MLP_Net(g, args).to(device)
        elif args.model == "JKNet":
            model = GCN_JKNet(g, args).to(device)
        elif args.model == "SGC":
            model = SGC_Net(g, args).to(device)
        elif args.model == "GCN-Cheby":
            model = Cheby_Net(g, args).to(device)
        elif args.model == "SAGE":
            model = SAGE_Net(g, args).to(device)
    model.reset_parameters()

###################
# optimizer select
###################
    if args.model in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam(
        [
        {
            'params':model.lin1.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
            {
            'params':model.lin2.parameters(),
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
            {
            'params':model.prop1.parameters(),
            'lr': args.lr,
            'weight_decay': 0.0
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)




    test_acc_list = []
    val_acc_list = []
    gamma_list = []

    for RP in tqdm(range(10)):
        test_acc, best_val_acc, Gamma_0 = RunExp(
                args, g, model, optimizer)
        test_acc_list.append(test_acc)
        val_acc_list.append(best_val_acc)
        gamma_list.append(Gamma_0)
    test_acc_mean = np.mean(test_acc_list, axis=0) * 100 
    val_acc_mean = np.mean(val_acc_list, axis=0) * 100 
    gamma_mean = np.mean(gamma_list, axis=0)
    test_acc_std = np.sqrt(np.var(test_acc_list)) * 100
    gamma_std = np.sqrt(np.var(gamma_list, axis=0)) 

    # test_acc_std = np.sqrt(np.var(gamma_list)
    # print(test_acc_list)
    # print(test_acc_std)
    # test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100








    if args.record == True:
        if args.model == "GPRGNN" and args.passing_type == "Attention":
            rfile = args.dataset + "_learnt_weight"+ ".txt"
            tfile = args.dataset + ".txt"
        else:
            # rfile = args.dataset + "_" + args.model + "_" + str(args.lr) + "_" + str(args.weight_decay) +"_record"+ ".txt"
            rfile = "fuck" + ".txt"
            tfile = args.dataset + ".txt"
        
        if args.train_ratio < 0.5:
            record_path_1 = osp.abspath("../result/final/sparse")
        else:
            record_path_1 = osp.abspath("../result/final/dense")
        rfile = osp.join(record_path_1, rfile)
        tfile = osp.join(record_path_1, tfile)
        # if os.path.exists(rfile):
        #     os.remove(rfile)
        # with open(rfile, "w") as f:
        #     f.write("epoch loss train_acc val_acc test_acc best_acc\n")
        #     for e in range(args.epoch):
        #         f.write("{} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(e+1, loss_list[e], train_acc_list[e], val_acc_list[e], test_acc_list[e], best_test_acc_list[e]))
        #     if args.model == "GPRGNN":
        #         f.write("{}".format(model.prop1.temp.detach().cpu().numpy()))
        
        with open(rfile, "a+") as f:
            f.write("{} {:.3f} {:.3f}".format(args.id, test_acc_mean.astype(float), test_acc_std.astype(float)))
            f.write("\n       ")
            for i in gamma_mean.tolist():
                f.write(" {:.3f}".format(i))
            f.write("\n       ")
            for i in gamma_std:
                f.write(" {:.3f}".format(i))
            f.write("\n")


        with open(tfile, "a+") as f:
            if args.model == "GPRGNN":
                m = args.model + " " +args.passing_type + " " + str(args.lr) + " " + str(args.weight_decay) + " " + str(args.hidden) + " " + str(args.dprate) + " " + str(args.alpha)
            else:
                m = args.model + " " + str(args.lr) + str(args.weight_decay) + " " + str(args.hidden) 
            f.write("{} {:.2f} {:.2f} {}\n".format(args.id, test_acc_mean, val_acc_mean, m))
        


