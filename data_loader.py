import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from config import Config

def load_data():

    cfg = Config()
    # QTs
    # M = np.loadtxt(f"{cfg.data_dir}/116_adj/adni_{cfg.image}.txt", 
    #               dtype=float, delimiter='\t')
    
    # label
    # Y = np.loadtxt(f"{cfg.data_dir}/adni_diag.txt", 
    #               dtype=float, delimiter='\t')
    
    M=np.random.rand(100,10)
    Y=np.random.rand(100,10)
    
    M = torch.from_numpy(M).to(dtype=cfg.dtype, device=cfg.device)
    Y = torch.from_numpy(Y).to(dtype=cfg.dtype, device=cfg.device)
    # genotype data
    # with h5py.File(f"{cfg.data_dir}/adni_36.h5", 'r') as hdf5_file:
    #     genotype_data = hdf5_file['genotype_data'][:]
    
    genotype_data=np.random.rand(100,1)
    genotype_data = torch.from_numpy(genotype_data).to(dtype=cfg.dtype,device=cfg.device)
    # split data
    indices = np.arange(Y.shape[0])
    train_idx, test_idx = train_test_split(indices, 
                                     test_size=cfg.test_size ,
                                     random_state=cfg.random_state)
    return {
        'M': M, 'Y': Y, 'genotype_data': genotype_data,
        'train_index': train_idx, 'test_index': test_idx
    }