import numpy as np
import torch

class Config:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    max_iter = 20
    tol = 1e-4
    chunk_size = 50000
    
    # Hyperparameter grid
    # example
    lamda_1_set = np.array([0.001,0.005,0.01,0.05])
    lamda_2_set = np.array([0.001,0.005,0.01,0.05])
    lamda_3_set = np.array([0.001,0.005,0.01,0.05])
    lamda_4_set = np.array([0.001,0.005,0.01,0.05])
    rho_set = np.array([0.001,0.005,0.01,0.05])
    
    len_1=len(lamda_1_set)*len(lamda_2_set)*len(lamda_3_set)*len(lamda_4_set)*len(rho_set)
    len_2=len(lamda_2_set)*len(lamda_3_set)*len(lamda_4_set)*len(rho_set)
    len_3=len(lamda_3_set)*len(lamda_4_set)*len(rho_set)
    len_4=len(lamda_4_set)*len(rho_set)
    # Path configuration
    data_dir = 'C:/Users/DELL/Desktop/multi_task/data'
    result_dir = 'D:/yy/median/116_adj'
    image = 'fdg'
    
    # split data 
    test_size = 0.2
    n_splits = 5
    random_state = 42