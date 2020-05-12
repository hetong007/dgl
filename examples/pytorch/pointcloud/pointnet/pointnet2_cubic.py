import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv

'''
class FixedRadiusNNGraph(nn.Module):
    def __init__(self, radius, n_neighbor):
        super(FixedRadiusNNGraph, self).__init__()
        self.radius = radius
        self.n_neighbor = n_neighbor
        self.frnn = FixedRadiusNearNeighbors(radius, n_neighbor)

    def forward(self, pos, centroids, feat=None):
        dev = pos.device
        group_idx = self.frnn(pos, centroids)
        B, N, _ = pos.shape
        glist = []
        for i in range(B):
            center = torch.zeros((N)).to(dev)
            center[centroids[i]] = 1
            src = group_idx[i].contiguous().view(-1)
            dst = centroids[i].view(-1, 1).repeat(1, self.n_neighbor).view(-1)

            unified = torch.cat([src, dst])
            uniq, idx, inv_idx = np.unique(unified.cpu().numpy(), return_index=True, return_inverse=True)
            src_idx = inv_idx[:src.shape[0]]
            dst_idx = inv_idx[src.shape[0]:]

            g = dgl.DGLGraph((src_idx, dst_idx), readonly=True)
            g.ndata['pos'] = pos[i][uniq]
            g.ndata['center'] = center[uniq]
            if feat is not None:
                g.ndata['feat'] = feat[i][uniq]
            glist.append(g)
        bg = dgl.batch(glist)
        return bg
'''

class Point2Voxel(nn.Module):
    def __init__(self, radius):
        super(Point2Voxel, self).__init__()
        self.radius = radius

    def extract_bin(self, D):
        uplimit = 3 ** D
        binstr = [int(np.base_repr(x, base=3)) for x in range(uplimit)]
        formatstr = '%0' + str(D) + 'd'
        bin_map = [[int(x) - 1 for x in list(formatstr%(bs))] for bs in binstr]
        return np.array(bin_map)

    def forward(self, points):
        B, N, D = points.shape
        bin_map = self.extract_bin(D)
        ht = [{} for i in range(B)]
        keys = (points // radius).astype(np.int).tolist()
        for i in range(B):
            for j in range(N):
                key = tuple(keys[i][j])
                if key not in ht[i]:
                    ht[i][key] = {'ind': [j], 'n_center': 0, 'n_src': 0, 'is_center': [0], 'is_src': [0]}
                else:
                    ht[i][key]['ind'].append(j)
                    ht[i][key]['is_center'].append(0)
                    ht[i][key]['is_src'].append(0)
        for i in range(B):
            for k, v in ht[i].items():
                v['ind'] = np.array(v['ind'])
                v['is_center'] = np.array(v['is_center'])
                v['is_src'] = np.array(v['is_src'])
        return keys, ht, bin_map

class GridNNGraph(nn.Module):
    def __init__(self, radius, n_sample, n_neighbor):
        super(GridNNGraph, self).__init__()
        self.radius = radius
        self.n_sample = n_sample
        self.n_neighbor = n_neighbor
        self.p2v = Point2Voxel(radius)

    def extract_neighbor_keys(self, key, bin_map):
        bucket_ind = np.array(key)
        neighbor_keys = (bucket_ind + bin_map).tolist()
        return neighbor_keys

    def forward(self, pos, feat=None):
        B, N, D = pos.shape
        keys, ht, bin_map = self.p2v(pos)
        stat = [{} for i in range(B)]
        center_flag = np.zeros((B, N))
        src_flag = np.zeros((B, N))
        sample = np.zeros((B, self.n_sample))

        glist = []
        for i in range(B):
            src_list = []
            dst_list = []
            for t in range(n_sample):
                key = ht[i].keys()

                # sample one voxel
                prob_vec = [len(v['ind']) / (v['n_src'] + 1) + len(v['ind']) * self.n_neighbor / (v['n_center'] + 1) for k, v in ht[i].items()]
                voxel_ind = np.random.choice(len(ht[i]), size=1, p=prob_vec).tolist()[0]
                center_key = key[voxel_ind]

                # sample point from one voxel
                prob_vec = (1 - ht[i][center_key]['is_center']) * (1 - ht[i][center_key]['is_src'])
                point_ind = np.random.choice(len(ht[i][center_key]['ind']), size=1, p=prob_vec).tolist()[0]
                point_global_ind = ht[i][center_key]['ind']

                # sample the center's src
                neighbor_keys = self.extract_neighbor_keys(k, bin_map)
                neighbor_points = []
                point_key_mapping = []
                prob_vec = []
                for k in neighbor_keys:
                    if k in key:
                        point_key_mapping += list(range(len(ht[i][k]['ind'])))
                        neighbor_points += ht[i][k]['ind']
                src_ind = np.random.choice(neighbor_points, size=self.n_neighbor)

                # update src/dst
                src_list += src_ind
                dst_list += [point_ind for i in range(self.n_neighbor)]

                # update stats
                center_flag[i, point_global_ind] = 1
                ht[i][center_key]['is_center'][point_ind] = 1
                src_keys = keys[src_ind]
                for j in range(self.n_neighbor):
                    ind = src_ind[j]
                    src_key = keys[nd]
                    src_ind = point_key_mapping[ind]
                    ht[i][src_key]['is_src'] = 1
                
                for k, v in ht[i].items():
                    v['n_center'] = v['is_center'].sum()
                    v['n_src'] = v['is_src'].sum()

            g = dgl.DGLGraph((src_idx, dst_idx), readonly=True)
            g.ndata['pos'] = pos[i]
            g.ndata['center'] = center_flag[i]
            if feat is not None:
                g.ndata['feat'] = feat[i]
            glist.append(g)
        bg = dgl.batch(glist)
        return bg
        
class GroupMessage(nn.Module):
    '''
    Compute the input feature from neighbors
    '''
    def __init__(self, n_neighbor):
        super(GroupMessage, self).__init__()
        self.n_neighbor = n_neighbor

    def forward(self, edges):
        pos = edges.src['pos'] - edges.dst['pos']
        if 'feat' in edges.src:
            res = torch.cat([pos, edges.src['feat']], 1)
        else:
            res = pos
        return {'agg_feat': res}

class PointNetConv(nn.Module):
    '''
    Feature aggregation
    '''
    def __init__(self, sizes, batch_size):
        super(PointNetConv, self).__init__()
        self.batch_size = batch_size
        self.conv = nn.ModuleList()
        self.bn = nn.ModuleList()
        for i in range(1, len(sizes)):
            self.conv.append(nn.Conv2d(sizes[i-1], sizes[i], 1))
            self.bn.append(nn.BatchNorm2d(sizes[i]))

    def forward(self, nodes):
        shape = nodes.mailbox['agg_feat'].shape
        h = nodes.mailbox['agg_feat'].view(self.batch_size, -1, shape[1], shape[2]).permute(0, 3, 1, 2)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h, 3)[0]
        feat_dim = h.shape[1]
        h = h.permute(0, 2, 1).reshape(-1, feat_dim)
        return {'new_feat': h}
    
    def group_all(self, pos, feat):
        '''
        Feature aggretation and pooling for the non-sampling layer
        '''
        if feat is not None:
            h = torch.cat([pos, feat], 2)
        else:
            h = pos
        shape = h.shape
        h = h.permute(0, 2, 1).view(shape[0], shape[2], shape[1], 1)
        for conv, bn in zip(self.conv, self.bn):
            h = conv(h)
            h = bn(h)
            h = F.relu(h)
        h = torch.max(h[:, :, :, 0], 2)[0]
        return h

class SAModule(nn.Module):
    def __init__(self, npoints, batch_size, radius, mlp_sizes, n_neighbor=64,
                 group_all=False):
        super(SAModule, self).__init__()
        self.group_all = group_all
        if not group_all:
            self.fps = FarthestPointSampler(npoints)
            self.frnn_graph = FixedRadiusNNGraph(radius, n_neighbor)
        self.message = GroupMessage(n_neighbor)
        self.conv = PointNetConv(mlp_sizes, batch_size)
        self.batch_size = batch_size

    def forward(self, pos, feat):
        if self.group_all:
            return self.conv.group_all(pos, feat)

        centroids = self.fps(pos)
        g = self.frnn_graph(pos, centroids, feat)
        g.update_all(self.message, self.conv)
        mask = g.ndata['center'] == 1
        pos_dim = g.ndata['pos'].shape[-1]
        feat_dim = g.ndata['new_feat'].shape[-1]
        pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
        feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
        return pos_res, feat_res

class SAMSGModule(nn.Module):
    def __init__(self, npoints, batch_size, radius_list, n_neighbor_list, mlp_sizes_list):
        super(SAMSGModule, self).__init__()
        self.batch_size = batch_size
        self.group_size = len(radius_list)

        self.fps = FarthestPointSampler(npoints)
        self.frnn_graph_list = nn.ModuleList()
        self.message_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        for i in range(self.group_size):
            self.frnn_graph_list.append(FixedRadiusNNGraph(radius_list[i],
                                                           n_neighbor_list[i]))
            self.message_list.append(GroupMessage(n_neighbor_list[i]))
            self.conv_list.append(PointNetConv(mlp_sizes_list[i], batch_size))

    def forward(self, pos, feat):
        centroids = self.fps(pos)
        feat_res_list = []
        for i in range(self.group_size):
            g = self.frnn_graph_list[i](pos, centroids, feat)
            g.update_all(self.message_list[i], self.conv_list[i])
            mask = g.ndata['center'] == 1
            pos_dim = g.ndata['pos'].shape[-1]
            feat_dim = g.ndata['new_feat'].shape[-1]
            if i == 0:
                pos_res = g.ndata['pos'][mask].view(self.batch_size, -1, pos_dim)
            feat_res = g.ndata['new_feat'][mask].view(self.batch_size, -1, feat_dim)
            feat_res_list.append(feat_res)
        feat_res = torch.cat(feat_res_list, 2)
        return pos_res, feat_res

class PointNet2SSGCls(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(PointNet2SSGCls, self).__init__()
        self.input_dims = input_dims

        self.sa_module1 = SAModule(512, batch_size, 0.2, [input_dims, 64, 64, 128])
        self.sa_module2 = SAModule(128, batch_size, 0.4, [128 + 3, 128, 128, 256])
        self.sa_module3 = SAModule(None, batch_size, None, [256 + 3, 256, 512, 1024],
                                   group_all=True)

        self.mlp1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_prob)

        self.mlp2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_prob)

        self.mlp_out = nn.Linear(256, output_classes)

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        pos, feat = self.sa_module1(pos, feat)
        pos, feat = self.sa_module2(pos, feat)
        h = self.sa_module3(pos, feat)

        h = self.mlp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.mlp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)

        out = self.mlp_out(h)
        return out

class PointNet2MSGCls(nn.Module):
    def __init__(self, output_classes, batch_size, input_dims=3, dropout_prob=0.4):
        super(PointNet2MSGCls, self).__init__()
        self.input_dims = input_dims

        self.sa_msg_module1 = SAMSGModule(512, batch_size, [0.1, 0.2, 0.4], [16, 32, 128],
                                          [[input_dims, 32, 32, 64], [input_dims, 64, 64, 128],
                                           [input_dims, 64, 96, 128]])
        self.sa_msg_module2 = SAMSGModule(128, batch_size, [0.2, 0.4, 0.8], [32, 64, 128],
                                          [[320 + 3, 64, 64, 128], [320 + 3, 128, 128, 256],
                                           [320 + 3, 128, 128, 256]])
        self.sa_module3 = SAModule(None, batch_size, None, [640 + 3, 256, 512, 1024],
                                   group_all=True)

        self.mlp1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(dropout_prob)

        self.mlp2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(dropout_prob)

        self.mlp_out = nn.Linear(256, output_classes)

    def forward(self, x):
        if x.shape[-1] > 3:
            pos = x[:, :, :3]
            feat = x[:, :, 3:]
        else:
            pos = x
            feat = None
        pos, feat = self.sa_msg_module1(pos, feat)
        pos, feat = self.sa_msg_module2(pos, feat)
        h = self.sa_module3(pos, feat)

        h = self.mlp1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.drop1(h)
        h = self.mlp2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = self.drop2(h)

        out = self.mlp_out(h)
        return out
