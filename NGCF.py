import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化LSTM的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 通过LSTM层
        out, _ = self.lstm(x, (h0, c0))

        # 只使用最后一个时间步的输出
        out = out[:, -1, :]

        # 通过全连接层得到最终输出
        out = self.fc(out)

        return out

class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, CircEmbeddingFeature, miRNAEmbeddingFeature, args):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj

        self.CircEmbeddingFeature= CircEmbeddingFeature
        self.miRNAEmbeddingFeature= miRNAEmbeddingFeature

        self.layers = eval(args.layer_size)

        self.circ_RNN = LSTMModel(input_size = 64, hidden_size = 256, num_layers = 2, output_size = 128)  # 84+30+768
        self.miR_RNN = LSTMModel(input_size = 64, hidden_size = 256, num_layers = 2, output_size = 128)  # 84+30+768

        self.weight_dict = self.init_weight()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_


        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        # return embedding_dict, weight_dict
        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def forward(self, users, items, drop_flag=False):

        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if drop_flag else self.sparse_norm_adj


        self.embedding_dict = nn.ParameterDict({
                'user_emb': self.miR_RNN(self.miRNAEmbeddingFeature),
                'item_emb': self.circ_RNN(self.CircEmbeddingFeature)
        })

        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            side_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            sum_embeddings = torch.matmul(side_embeddings, self.weight_dict['W_gc_%d' % k]) \
                                             + self.weight_dict['b_gc_%d' % k]

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = torch.matmul(bi_embeddings, self.weight_dict['W_bi_%d' % k]) \
                                            + self.weight_dict['b_bi_%d' % k]

            ego_embeddings = nn.LeakyReLU(negative_slope=0.2)(sum_embeddings + bi_embeddings)
            ego_embeddings = nn.Dropout(self.mess_dropout[k])(ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings = all_embeddings[:self.n_user, :]
        i_g_embeddings = all_embeddings[self.n_user:, :]

        u_g_embeddings = u_g_embeddings / u_g_embeddings.norm(p=2, dim=1, keepdim=True)
        i_g_embeddings = i_g_embeddings / i_g_embeddings.norm(p=2, dim=1, keepdim=True)

        u_g_embeddings = u_g_embeddings[users, :]
        i_g_embeddings = i_g_embeddings[items, :]

        return u_g_embeddings, i_g_embeddings
