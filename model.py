import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import args


class AVGAE(nn.Module):
    def __init__(self, adj):
        super(AVGAE, self).__init__()
        self.base_gcn = GraphAttnLayer(
            args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphAttnLayer(
            args.hidden1_dim, args.hidden2_dim, adj)
        self.gcn_logstddev = GraphAttnLayer(
            args.hidden1_dim, args.hidden2_dim, adj)
        self.adj = adj

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class VGAE(nn.Module):
    def __init__(self, adj):
        super(VGAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(
            args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)
        self.gcn_logstddev = SpGAT(
            args.hidden1_dim, args.hidden2_dim, args.hidden2_dim, adj)

    def encode(self, X):
        hidden = self.base_gcn(X)
        self.mean = self.gcn_mean(hidden)
        self.logstd = self.gcn_logstddev(hidden)
        gaussian_noise = torch.randn(X.size(0), args.hidden2_dim)
        sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
        return sampled_z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


class GraphAttnLayer(nn.Module):
    def __init__(self, input_dim, output_dim, adj, num_heads=1, dropout=0.4, alpha=0.2, activation=F.relu, **kwargs):
        super(GraphAttnLayer, self).__init__(**kwargs)
        self.output_dim = output_dim//num_heads
        self.weights = [glorot_init(input_dim, self.output_dim)
                        for _ in range(num_heads)]
        self.adj = adj
        self.dropout = dropout
        self.input_dim = input_dim
        self.alpha = alpha
        self.a = nn.Parameter(torch.zeros(size=(2*(self.output_dim), 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, X):
        # import pdb
        # pdb.set_trace()
        h_s = [torch.mm(X, self.weights[i]) for i in range(len(self.weights))]
        N = h_s[0].size(0)

        a_s = [torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)],
                         dim=1).view(N, -1, 2 * self.output_dim) for h in h_s]
        e_s = [self.leakyrelu(torch.matmul(a, self.a).squeeze(2)) for a in a_s]

        mask = -9e15 * torch.ones_like(e_s[0])
        attn = [torch.where(self.adj.to_dense() > 0, e, mask) for e in e_s]
        attn = [F.softmax(att, dim=1) for att in attn]
        attn = [F.dropout(att, self.dropout, training=self.training)
                for att in attn]
        h_s_prime = torch.cat([torch.matmul(att, h)
                              for (att, h) in zip(attn, h_s)], dim=1)

        return h_s_prime


class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, adj, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(self.adj, x)
        outputs = self.activation(x)
        return outputs


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)


class GAE(nn.Module):
    def __init__(self, adj):
        super(GAE, self).__init__()
        self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
        self.gcn_mean = GraphConvSparse(
            args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x: x)

    def encode(self, X):
        hidden = self.base_gcn(X)
        z = self.mean = self.gcn_mean(hidden)
        return z

    def forward(self, X):
        Z = self.encode(X)
        A_pred = dot_product_decode(Z)
        return A_pred


# class GraphConv(nn.Module):
# 	def __init__(self, input_dim, hidden_dim, output_dim):
# 		super(VGAE,self).__init__()
# 		self.base_gcn = GraphConvSparse(args.input_dim, args.hidden1_dim, adj)
# 		self.gcn_mean = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)
# 		self.gcn_logstddev = GraphConvSparse(args.hidden1_dim, args.hidden2_dim, adj, activation=lambda x:x)

# 	def forward(self, X, A):
# 		out = A*X*self.w0
# 		out = F.relu(out)
# 		out = A*X*self.w0
# 		return out


class MyGAT(nn.Module):
    def __init__(self):
        super(MyGAT, self).__init__()


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout=0, alpha=0.1, nheads=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.adj = adj

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, adj, dropout=0, alpha=0.1, nheads=2):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)
        self.adj = adj

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return F.log_softmax(x, dim=1)
