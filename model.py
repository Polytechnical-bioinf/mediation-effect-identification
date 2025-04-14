import torch
from config import Config

class MediationModel:
    def __init__(self, num_snps, n_features, n_tasks):
        cfg = Config()
        self.beta_ = torch.zeros((n_tasks, num_snps), 
                               dtype=cfg.dtype, device=cfg.device)
        self.W2 = torch.zeros((n_tasks, n_features), 
                            dtype=cfg.dtype, device=cfg.device)
        self.W1 = torch.zeros((n_features, num_snps), 
                            dtype=cfg.dtype, device=cfg.device)
        self.beta = torch.zeros((n_tasks, num_snps), 
                             dtype=cfg.dtype, device=cfg.device)
    
    def reset_parameters(self):

        self.beta_.zero_()
        self.W2.zero_()
        self.W1.zero_()
        self.beta.zero_()