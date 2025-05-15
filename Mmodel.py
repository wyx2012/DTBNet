import numpy as np
import torch
import torch_geometric
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from graph_model.GAT import GAT
from graph_model.EGAT import EGNNConv3D
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

class Mmodel(torch.nn.Module):
    def __init__(self):
        super(Mmodel, self).__init__()
        d_vocab = 21
        d_embed = 20
        d_dihedrals = 6
        d_pretrained_emb = 1280
        d_edge = 39
        d_gcn = [128, 256, 256]
        d_gcn_in = d_gcn[0]
        self.protein_data = torch.load('./data/protein_data.pt')
        self.drug_data = torch.load('./data/drug_data.pt',weights_only=False)
        self.l_node = nn.Linear(d_pretrained_emb+d_embed+d_dihedrals+d_embed, d_gcn_in)
        self.l_edge = nn.Linear(d_edge, d_gcn_in)
        gcn_layer_sizes = [d_gcn_in] + d_gcn
        layers = []
        for i in range(len(gcn_layer_sizes) - 1):
            layers.append((
                torch_geometric.nn.TransformerConv(
                    gcn_layer_sizes[i], gcn_layer_sizes[i + 1], edge_dim=d_gcn_in),
                'x, edge_index, edge_attr -> x'
            ))
            layers.append(nn.LeakyReLU())
        self.gcn = torch_geometric.nn.Sequential(
            'x, edge_index, edge_attr', layers)
        self.bert_mpl = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.mlp_ecfp = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.mlp_hash = nn.Sequential(
            nn.Linear(1024, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.GraphDrug = EGNNConv3D(78, 78)
        self.resmodel = models.resnet18(pretrained=True)
        self.resmodel = torch.nn.Sequential(*(list(self.resmodel.children())[:-1]))
        #%% 特征融合模块---------------------------------------------------------
        # 对齐模块
        self.e_proj = nn.Sequential(
            nn.Conv1d(30, 16, kernel_size=3, padding=1),  # [b,16,78]
            nn.AdaptiveAvgPool1d(16),  # [b,16,16]
            nn.Flatten(),
            nn.Linear(16 * 16, 256)
        )
        self.f_proj = nn.Linear(512, 256)
        # 相似特征分组融合
        self.group1_fusion = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        self.group2_fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        self.group3_fusion = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        # 跨组交互模块
        self.cross_attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )







    def forward(self,smiles,protein,label,batchsize):
        #%% protein_start
        features_p_edges_list = []
        features_p_edge_list = []
        features_p_emb_list = []
        features_p_bert_list = []
        p_graph = []
        for protein_id in protein:
            pid = protein_id.item()
            protein_tuple = self.protein_data[pid] #node_s,edge_s,edge_index,seq_emb,bert_emb,seq_emb2,one_hot
            features_p_node_s = protein_tuple[0].cuda(0)
            features_p_edges = protein_tuple[1].cuda(0)
            features_p_edge = protein_tuple[2].cuda(0)
            features_p_emb = protein_tuple[3].cuda(0)
            feature_p_bert = protein_tuple[4].cuda(0)
            feature_p_emb2 = protein_tuple[5].cuda(0)
            feature_p_one_hot = protein_tuple[6].cuda(0)



            x_emb =torch.cat([features_p_emb, features_p_node_s, feature_p_emb2,feature_p_one_hot], dim=-1)
            x_emb = self.l_node(x_emb)
            edge_attr = self.l_edge(features_p_edges)
            x = self.gcn(x_emb, features_p_edge, edge_attr)
            x = torch.mean(x, dim=0, keepdim=True)
            p_graph.append(x)
            features_p_edges_list.append(features_p_edges)
            features_p_edge_list.append(features_p_edge)
            features_p_emb_list.append(features_p_emb)
            features_p_bert_list.append(feature_p_bert)
        p_graph = torch.stack(p_graph, dim=0).squeeze()
        p_bert = torch.cat(features_p_bert_list, dim=0)
        p_bert = self.bert_mpl(p_bert)




        #%% drug_start---------------药物-----------------------

        features_d_img_list = []
        features_d_ecfp_list = []
        features_d_hash_list = []
        features_d_feature_list = []
        features_d_e_list = []
        features_d_p_list = []
        d_graph = []
        for smiles_id in smiles:
            did = smiles_id.item()
            drug_tuple = self.drug_data[did]
            features_d_img = torch.tensor(drug_tuple[0])
            features_d_ecfp = drug_tuple[1]
            features_d_hash = drug_tuple[2]
            features_d_feature = drug_tuple[3].float().cuda(0)#float64
            features_d_e = drug_tuple[4].cuda(0)
            features_d_p = drug_tuple[5].cuda(0)
            mask = ~(features_d_e[:, 0] == 0) & ~(features_d_e[:, 1] == 0)
            features_d_e = features_d_e[mask]
            features_d_e = features_d_e.t().int()
            features_d_e = torch.where(features_d_e > 29, torch.zeros_like(features_d_e), features_d_e)
            d_gat = self.GraphDrug(features_d_feature, features_d_p, features_d_e)
            d_graph.append(d_gat)

            features_d_img_list.append(features_d_img)
            features_d_ecfp_list.append(features_d_ecfp)
            features_d_hash_list.append(features_d_hash)
        d_graph = torch.stack(d_graph, dim=0)
        features_d_img_list = torch.stack(features_d_img_list).permute(0, 3, 1, 2)
        d_img = self.resmodel(features_d_img_list.float().cuda(0))
        d_img = d_img.squeeze()
        d_ecfp = torch.tensor(features_d_ecfp_list)
        d_hash = torch.tensor(features_d_hash_list)
        d_ecfp = self.mlp_ecfp(d_ecfp.float().cuda(0))
        d_hash = self.mlp_hash(d_hash.float().cuda(0))
        #%%特征融合--------------------------------------------------------------
        # 特征对齐
        d_graph = self.e_proj(d_graph.cuda(0))  # [b,30,78] -> [b,256]
        d_img = self.f_proj(d_img)

        # 分组特征融合
        group1 = self.group1_fusion(torch.cat([d_ecfp,p_bert, d_hash], dim=-1))  # 相似特征组1
        group2 = self.group2_fusion(torch.cat([p_graph.cuda(0),  d_img], dim=-1))  # 相似特征组2
        group3 = self.group3_fusion(torch.cat([p_graph.cuda(0), d_graph], dim=-1))

        # 跨组注意力交互
        cross_feat, _ = self.cross_attention(
            group1.unsqueeze(1),
            group2.unsqueeze(1),
            group3.unsqueeze(1)
        )
        combined = torch.cat([
            group1 + cross_feat.squeeze(1),
            group2 + group1.unsqueeze(1).mean(dim=1)
        ], dim=-1)
        # combined = torch.cat([
        #     cross_feat.squeeze(1),
        #     group1.unsqueeze(1).mean(dim=1)
        # ], dim=-1)


        return self.regressor(combined).squeeze()
