import torch
import torch.nn as nn

# class AttentionModel(nn.Module):
#     def __init__(self, input_size):
#         super(AttentionModel, self).__init__()
#         self.input_size = input_size
#         self.sigmoid = nn.Sigmoid()
#         self.Wq = nn.Parameter(nn.init.xavier_normal_(torch.empty((input_size, input_size))) / 100)
#         self.Wk = nn.Parameter(nn.init.xavier_normal_(torch.empty((input_size, input_size)))/ 100)
#
#
#     def forward(self, x):
#         x = x.unsqueeze(-1)
#         q = torch.einsum('lj,ijk->ilk', self.Wq, x)
#         k = torch.einsum('lj,ijk->ilk', self.Wk, x)
#         att_scores = torch.einsum('ijk,ilk->ijl', q, k)
#         att_scores = (self.sigmoid((att_scores / (self.input_size ** .5))).sum(1))
#         weighted_sum = (x.squeeze(-1) * att_scores).sum(1)
#         output = self.sigmoid(weighted_sum.unsqueeze(-1))
#
#         return output

# The MILCA Model:
class SigmoidWeightLogisticRegression(nn.Module):
    def __init__(self, input_size, drop_out):
        super(SigmoidWeightLogisticRegression, self).__init__()
        self.input_size = input_size
        self.sigmoid = nn.Sigmoid()
        self.Wq = nn.Parameter(nn.init.xavier_normal_(torch.empty((input_size, 1))))
        self.dropout = nn.Dropout(p=drop_out)
        self.mu = nn.Parameter(-1 / 2 * torch.ones((1, 1)))

    def forward(self, x):
        x = self.dropout(x)
        W = self.sigmoid(self.Wq) + self.mu
        att_scores = torch.matmul(x, W)
        return att_scores