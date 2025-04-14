import torch
from data_loader import load_data
from model import MediationModel
from mediation import Mediation
from config import Config
import numpy as np
from utils import define_matrix
def Cross_validate():
    cfg = Config()
    
    # load data
    data = load_data()
    num_snps = data['genotype_data'].shape[1]
    n_features = data['M'].shape[1]
    n_tasks = data['Y'].shape[1]
    mse_train=np.zeros((cfg.len_1,11))
    r2_train=np.zeros((cfg.len_1,11))

    mse_val=np.zeros((cfg.len_1,11))
    r2_val=np.zeros((cfg.len_1,11))

    model = MediationModel(num_snps, n_features, n_tasks)
    trainer = Mediation(model, data)
    #Grid search
    for i_lamda_1 in range(len(cfg.lamda_1_set)):
        lamda_1=cfg.lamda_1_set[i_lamda_1]
        for i_lamda_2 in range(len(cfg.lamda_2_set)):
            lamda_2=cfg.lamda_2_set[i_lamda_2]
            for i_lamda_3 in range(len(cfg.lamda_3_set)):
                lamda_3=cfg.lamda_3_set[i_lamda_3]
                for i_lamda_4 in range(len(cfg.lamda_4_set)):
                    lamda_4=cfg.lamda_4_set[i_lamda_4]
                    for i_rho in range(len(cfg.rho_set)):
                        
                        rho=cfg.rho_set[i_rho]
                        ind_i=i_lamda_1*cfg.len_2+i_lamda_2*cfg.len_3+i_lamda_3*cfg.len_4+i_lamda_4*len(cfg.rho_set)+i_rho
                        mse_train,r2_train,mse_val,r2_val=define_matrix(mse_train,r2_train,mse_val,r2_val,ind_i,lamda_1,lamda_2,lamda_3,lamda_4,rho)
                        mse_val,r2_val=trainer.train((lamda_1, lamda_2, lamda_3, lamda_4, rho,mse_train,r2_train,mse_val,r2_val,ind_i,n_tasks,num_snps,n_features))
    proper_ind_mse=np.argmin(mse_val[:,10])
    proper_ind_r2=np.argmax(r2_val[:,10])
    print('===============================================')
    print('ALL DONE')
    print('===============================================')
    print('The optimal parameter when MSE is used as an indicator:')
    print('lamda_1:',mse_val[proper_ind_mse,0],'\n','lamda_2:',mse_val[proper_ind_mse,1],'\n','lamda_3:',mse_val[proper_ind_mse,2],'\n','lamda_4:',mse_val[proper_ind_mse,3],'\n','lamda_rho:',mse_val[proper_ind_mse,4])
    print('MSE:',mse_val[proper_ind_mse,10],'\n')
    print('The optimal parameter when R2 is used as an indicator:')
    print('lamda_1:',r2_val[proper_ind_r2,0],'\n','lamda_2:',r2_val[proper_ind_r2,1],'\n','lamda_3:',r2_val[proper_ind_r2,2],'\n','lamda_4:',r2_val[proper_ind_r2,3],'\n','lamda_rho:',r2_val[proper_ind_r2,4])
    print('R2:',r2_val[proper_ind_r2,10])

if __name__ == "__main__":
    Cross_validate()