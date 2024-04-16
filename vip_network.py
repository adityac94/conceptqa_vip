import torch
import torch.nn as nn
import pdb
import numpy as np
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, query_size = 312, output_size=312, eps=None, batchnorm=False):
        super().__init__()
        self.query_size = query_size
        self.output_dim = output_size
        self.layer1 = nn.Linear(self.query_size, 2000)
        self.layer2 = nn.Linear(2000, 500)
        self.classifier = nn.Linear(500, self.output_dim)

        self.eps = eps
        self.current_max = 0

        if batchnorm:
            self.norm1 = torch.nn.BatchNorm1d(2000)
            self.norm2 = torch.nn.BatchNorm1d(500)
        else:
            self.norm1 = torch.nn.LayerNorm(2000)
            self.norm2 = torch.nn.LayerNorm(500)
        # activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, mask=None):
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))

        if self.eps == None:
         return self.classifier(x)

        else:
            query_logits = self.classifier(x)
            query_mask = torch.where(mask == 1, -1e9, 0.)
            query_logits = query_logits + query_mask.cuda()

            query = self.softmax(query_logits / self.eps)

            query = (self.softmax(query_logits / 1e-9) - query).detach() + query
            return query

    def change_eps(self, eps):
        self.eps = eps


class ConceptNet2(nn.Module):
    def __init__(self, embed_dims=512):
        super().__init__()
        self.embed_dims = embed_dims
        self.input_dim = self.embed_dims * 2

        # Architecture
        self.layer1 = nn.Linear(self.input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)

        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
        self.norm3 = nn.BatchNorm1d(128)
        self.norm4 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()

        self.head = nn.Linear(64, 1)

    def forward(self, x):
        #         img_emb = F.normalize(img_emb, p=2, dim=-1)
        #         txt_emb = F.normalize(txt_emb, p=2, dim=-1)

        # x = torch.hstack([img_emb, txt_emb])
        x = self.relu(self.norm1(self.layer1(x)))
        x = self.relu(self.norm2(self.layer2(x)))
        x = self.relu(self.norm3(self.layer3(x)))
        x = self.relu(self.norm4(self.layer4(x)))
        return self.head(x).squeeze()