import torch

def abs_max(W_col):
    return torch.max(torch.abs(W_col))

def diff_abs_max(W_col, w_ii):
    abs_diff = torch.abs(W_col - w_ii)
    return torch.max(abs_diff)


def define_matrix(mse_train,r2_train,mse_val,r2_val,ind_i,lamda_1,lamda_2,lamda_3,lamda_4,rho):
    mse_train[ind_i,0]=lamda_1
    mse_train[ind_i,1]=lamda_2
    mse_train[ind_i,2]=lamda_3
    mse_train[ind_i,3]=lamda_4
    mse_train[ind_i,4]=rho
    
    r2_train[ind_i,0]=lamda_1
    r2_train[ind_i,1]=lamda_2
    r2_train[ind_i,2]=lamda_3
    r2_train[ind_i,3]=lamda_4
    r2_train[ind_i,4]=rho
    
    mse_val[ind_i,0]=lamda_1
    mse_val[ind_i,1]=lamda_2
    mse_val[ind_i,2]=lamda_3
    mse_val[ind_i,3]=lamda_4
    mse_val[ind_i,4]=rho
    
    r2_val[ind_i,0]=lamda_1
    r2_val[ind_i,1]=lamda_2
    r2_val[ind_i,2]=lamda_3
    r2_val[ind_i,3]=lamda_4
    r2_val[ind_i,4]=rho
    return mse_train,r2_train,mse_val,r2_val