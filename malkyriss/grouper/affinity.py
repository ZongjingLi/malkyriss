'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-02 03:21:43
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-02 03:21:49
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from .backbone import ResidualDenseNetwork

class AffinityCalculator(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""
    
    @abstractmethod
    def calculate_entailment_logits(self, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""
    

class GeneralAffinityCalculator(AffinityCalculator):
    def __init__(self, name : str):
        super().__init__()
        self.name = name
        latent_dim = 128
        kq_dim = 32
        self.ks_map = nn.Linear(latent_dim, kq_dim)
        self.qs_map = nn.Linear(latent_dim, kq_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""
        return
    
    def calculate_entailment_logits(self, indices, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""
    
    def ideal_maps(self, annotated_masks):
        """
        input is:
        an dict annotated masks that contains a mask, is a key does not appear in the annotation, consider it's zero
        """
        if self.name in annotated_masks:return annotated_masks[self.name]
        else: return False

    def calculate_affinity_logits(self, indices, augument_features = None, device = None):
        _, B, N, K = indices.shape
        device = self.device if device is None else device

        if augument_features is not None:
            if "annotated_masks" in augument_features:
                features = self.ideal_maps(augument_features["annotated_masks"])
                if isinstance(features, bool): # for zero features, just return zero logits
                    print("calculated")
                    return torch.logit(torch.zeros([B,N,K])).to(device)
                #print(f"{B}x{N}xD", features.shape)
                features = features.permute(0,2,3,1)
                flatten_features = features.reshape(B,N,-1).to(device)
                flatten_ks = flatten_features
                flatten_qs = flatten_features
                B, N, D = flatten_features.shape
            else:
                features = augument_features["features"].reshape(B,N,-1).to(device)
                flatten_ks = self.ks_map(features)
                flatten_qs = self.qs_map(features)
                B, N, D = flatten_ks.shape

        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)

        # gather image features and flatten them into 1dim features
        x_features = torch.gather(flatten_ks, dim = 1, index = x_indices).reshape([B, N, K, D])
        y_features = torch.gather(flatten_qs, dim = 1, index = y_indices).reshape([B, N, K, D])

        B, N, K, D = x_features.shape
        
        x_features = x_features.reshape([B, N, K, D])
        y_features = y_features.reshape([B, N, K, D])

        if "annotated_masks" in augument_features:
            #logits = logit(x_features == y_features, eps = 1e-6)
            logits = torch.sum( ((x_features - y_features) ** 2) , dim = -1)** 0.5
            eps = 0.01
            #print(logits.max(), logits.min())
            inverse_div = 1 / ( eps + 17.2 * logits.reshape([B, N, K]) )
            #inverse_div = (logits < 0.1).float()
            logits = torch.logit(inverse_div)
            #print(inverse_div.shape, inverse_div.max(), inverse_div.min())
        else:
            logits = (x_features * y_features).sum(dim = -1) * (D ** -0.5)
        logits = logits.reshape([B, N, K])
        return logits
    

class ObjectAffinityCalculator(AffinityCalculator):
    def __init__(self, input_dim,  latent_dim):
        super().__init__()
        self.name : str = "object"
        kq_dim = 132
        #assert input_dim % 2 == 0,"input dim should be divisble by 2 as it is a pair of patches features"
        self.backbone = ResidualDenseNetwork(latent_dim, n_colors = input_dim)
        self.ks_map = nn.Linear(latent_dim, kq_dim)
        self.qs_map = nn.Linear(latent_dim, kq_dim)

    def calculate_affinity_feature(self, indices, img):
        _, B, N, K = indices.shape

        conv_features = self.backbone(img)
        conv_features = torch.nn.functional.normalize(conv_features, dim = -1, p = 2)
        conv_features = conv_features.permute(0,2,3,1)
        B, W, H, D = conv_features.shape

        flatten_features = conv_features.reshape([B,W*H,D])
        flatten_features = torch.cat([
            flatten_features, # [BxNxD]
            torch.zeros([B,1,D]), # [Bx1xD]
        ], dim = 1) # to add a pad feature to the edge case
        flatten_ks = self.ks_map(flatten_features)
        flatten_qs = self.qs_map(flatten_features)


        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)

        # gather image features and flatten them into 1dim features
        x_features = torch.gather(flatten_ks, dim = 1, index = x_indices)
        y_features = torch.gather(flatten_qs, dim = 1, index = y_indices)
        
        x_features = x_features.reshape([B, N, K, D])
        y_features = y_features.reshape([B, N, K, D])
        #rint(x_features.shape, y_features.shape)
        return torch.cat([x_features, y_features], dim = -1)
    
    def calculate_entailment_logits(self, logits_features, key = None):
        B, N, K, D = logits_features.shape
        assert D % 2 == 0, "not a valid feature dim"
        DC = D // 2
        x_features = logits_features[:,:,:,:DC]
        y_features = logits_features[:,:,:,DC:]
        logits = (x_features * y_features).sum(dim = -1) * (DC ** -0.5)
        logits = logits.reshape([B, N, K])
        return logits