from torch import nn
import torch
import torch.nn.functional as F
import numpy as np, yaml
from typing import List, Sequence
class LossFun(nn.Module):
    def __init__(self,
                 train_batch: int,
                 ms_multipliers: Sequence[float] = (0.5, 1.0, 2.0),
                 eps: float = 1e-8):
        super().__init__()
        self.train_batch = train_batch
        self.ms_multipliers = tuple(ms_multipliers)
        self.eps = eps

        with open("config.yaml") as f:
            raw_text = f.read()
        dataset = yaml.safe_load(raw_text)["dataset"]
        raw_text = raw_text.format(dataset=dataset)
        config = yaml.safe_load(raw_text)
        self.triplets_dis = np.load(str(config["path_triplets_truth"]))

    def _pair_distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.norm(x1 - x2, p=2, dim=1)


    def forward(self,
                embedding_a: torch.Tensor,
                embedding_p: torch.Tensor,
                embedding_n: torch.Tensor,
                batch_index: List[int]) -> torch.Tensor:

        gt = torch.as_tensor(self.triplets_dis[batch_index],
                             device=embedding_a.device,
                             dtype=embedding_a.dtype)            
        d_ap_gt, d_an_gt = gt[:, 0], gt[:, 1]                    

        d_ap = self._pair_distance(embedding_a, embedding_p)     
        d_an = self._pair_distance(embedding_a, embedding_n)      

        d_gt_all = torch.cat([d_ap_gt, d_an_gt], 0)               
        d_all    = torch.cat([d_ap,    d_an   ], 0)              

        sigma_base = d_gt_all.median().clamp(min=1e-3)
        sigmas = sigma_base * torch.tensor(self.ms_multipliers,
                                           device=d_gt_all.device,
                                           dtype=d_gt_all.dtype)  

        d2     = d_gt_all.pow(2).unsqueeze(1)                    
        sigma2 = sigmas.pow(2).unsqueeze(0)                       
        w      = torch.exp(- d2 / (2.0 * sigma2 + self.eps))      

        err2 = (d_all - d_gt_all).pow(2).unsqueeze(1)             
        num  = (w * err2).sum(dim=0)                              
        den  = w.sum(dim=0).clamp(min=self.eps)                   
        loss_per_scale = num / den                               

        return loss_per_scale.mean()  
