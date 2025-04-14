import time
import numpy as np
import torch
from sklearn.model_selection import KFold
from config import Config
from utils import *
from evaluator import Evaluator
import os

class Mediation:
    def __init__(self, model, data):
        self.cfg = Config()
        self.model = model
        self.data = data
        self.kf = KFold(n_splits=self.cfg.n_splits, 
                       shuffle=True, 
                       random_state=self.cfg.random_state)
        
    def train(self, lamda_params):
        self.lamda_1, self.lamda_2, self.lamda_3, self.lamda_4, self.rho,self.mse_train,self.r2_train,self.mse_val,self.r2_val ,self.ind_i,self.n_tasks,self.num_snps,self.n_features= lamda_params
        print('--------------------------------------------------------------')
        print('Training parameter:','lamda_1:',self.lamda_1,'\n','lamda_2:',self.lamda_2,'\n','lamda_3:',self.lamda_3,'\n','lamda_4:',self.lamda_4,'\n','lamda_rho:',self.rho)
        #The training set is further divided into validation set and training set
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.data['train_index'])):
            print(train_idx.shape)
            print('Fold:',fold+1)
            self.R1=self.data['Y'][train_idx,:]
            self.R2=self.data['Y'][train_idx,:]
            self.R3=self.data['M'][train_idx,:]
            #Reset weight
            self.model.reset_parameters()
            self.mse_train_min=10
            self.r2_train_max=0
            self.mse_val_min=10
            self.r2_val_max=0
            for n_iter in range(self.cfg.max_iter):
                print('n_iter：',n_iter+1)
                self.w_max = torch.zeros(1,device=self.cfg.device)
                self.d_w_max = torch.zeros(1,device=self.cfg.device)
                self.n_batch=int(np.ceil((self.num_snps)/self.cfg.chunk_size))
            
                self.loss_w1=1/torch.sum(self.R1[:,0].to(torch.float32)**2)
                self.loss_w2=1/torch.sum(self.R2[:,0].to(torch.float32)**2)
                self.loss_w3=1/torch.sum(self.R3 ** 2, dim=0)
                
                self._update_W2(train_idx)
                self._update_beta_(train_idx)
                self._update_beta(train_idx)
                self._update_W1( train_idx)
               
                w_max=max(self.w1_max,self.w2_max,self.beta_max,self._beta_max)
                d_w_max=max(self.d_w1_max,self.d_w2_max,self.d_beta_max,self._d_beta_max)

                if w_max == 0.0:
                    print('The weights are all 0')
                    break
                elif d_w_max < self.cfg.tol  or n_iter == self.cfg.max_iter - 1:
                    self._evaluate(train_idx, val_idx, fold)
                    #self._save_metrics(fold, n_iter)
                    break
        return self.mse_val,self.r2_val
    def test(self, lamda_1, lamda_2, lamda_3, lamda_4, rho,n_tasks,num_snps,n_features):
        self.lamda_1, self.lamda_2, self.lamda_3, self.lamda_4, self.rho,self.n_tasks,self.num_snps,self.n_features=lamda_1, lamda_2, lamda_3, lamda_4, rho,n_tasks,num_snps,n_features
        test_idx=self.data['test_index']
        train_idx=self.data['train_index']
        M_train = self.data['M'][train_idx,:]
        Y_train = self.data['Y'][train_idx,:]

        self.R1=Y_train
        self.R2=Y_train
        self.R3=M_train
        for n_iter in range(self.cfg.max_iter):
            print('n_iter：',n_iter+1)
            self.w_max = torch.zeros(1,device=self.cfg.device)
            self.d_w_max = torch.zeros(1,device=self.cfg.device)
            self.n_batch=int(np.ceil((self.num_snps)/self.cfg.chunk_size))
 
            self.loss_w1=1/torch.sum(self.R1[:,0].to(torch.float32)**2)
            self.loss_w2=1/torch.sum(self.R2[:,0].to(torch.float32)**2)
            self.loss_w3=1/torch.sum(self.R3 ** 2, dim=0)
            
            self._update_W2(train_idx)
            self._update_beta_(train_idx)
            self._update_beta(train_idx)
            self._update_W1( train_idx)
            
            w_max=max(self.w1_max,self.w2_max,self.beta_max,self._beta_max)
            d_w_max=max(self.d_w1_max,self.d_w2_max,self.d_beta_max,self._d_beta_max)
            
          
            if w_max == 0.0:
                print('The weights are all 0')
                self.mse_test,self.r2_test=0,0
                break
            elif d_w_max < self.cfg.tol  or n_iter == self.cfg.max_iter - 1:
                self._evaluate_test(train_idx, test_idx)
                break
        return self.mse_test,self.r2_test,self.mse_eq1,self.mse_eq2,self.mse_eq3,self.r2_eq1,self.r2_eq2,self.r2_eq3


    def _evaluate_test(self,train_idx, test_idx):

        #Training performance
        pre1=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.W1.t())
        pre2=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.beta.t())
        pre3=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.beta_.t())+torch.matmul(self.data['M'][train_idx,:],self.model.W2.t())

        mse_1,r2_1 = Evaluator.calculate_metrics(self.data['Y'][train_idx,:].cpu().numpy() ,pre2.cpu().numpy())
        mse_2,r2_2 = Evaluator.calculate_metrics(self.data['Y'][train_idx,:].cpu().numpy() ,pre3.cpu().numpy())
        mse_3,r2_3 = Evaluator.calculate_metrics(self.data['M'][train_idx,:].cpu().numpy() ,pre1.cpu().numpy())
        self.mse_train=(mse_1+mse_2+mse_3)/3
        self.r2_train=(r2_1+r2_2+r2_3)/3
        #Testing performance
        pre1=torch.matmul(self.data['genotype_data'][test_idx,:],self.model.W1.t())
        pre2=torch.matmul(self.data['genotype_data'][test_idx,:],self.model.beta.t())
        pre3=torch.matmul(self.data['genotype_data'][test_idx,:],self.model.beta_.t())+torch.matmul(self.data['M'][test_idx,:],self.model.W2.t())
        mse_1,r2_1 = Evaluator.calculate_metrics(self.data['Y'][test_idx,:].cpu().numpy() ,pre2.cpu().numpy())
        mse_2,r2_2 = Evaluator.calculate_metrics(self.data['Y'][test_idx,:].cpu().numpy() ,pre3.cpu().numpy())
        mse_3,r2_3 = Evaluator.calculate_metrics(self.data['M'][test_idx,:].cpu().numpy() ,pre1.cpu().numpy())
        self.mse_test=(mse_1+mse_2+mse_3)/3
        self.r2_test=(r2_1+r2_2+r2_3)/3
        self.mse_eq1=mse_1
        self.mse_eq2=mse_2
        self.mse_eq3=mse_3
        self.r2_eq1=r2_1
        self.r2_eq2=r2_2
        self.r2_eq3=r2_3
        #Save the result of the weight
        self._save_metrics()
      
    def _evaluate(self,train_idx, val_idx,fold):
        #Training performance
        pre1=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.W1.t())
        pre2=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.beta.t())
        pre3=torch.matmul(self.data['genotype_data'][train_idx,:],self.model.beta_.t())+torch.matmul(self.data['M'][train_idx,:],self.model.W2.t())

        mse_1,r2_1 = Evaluator.calculate_metrics(self.data['Y'][train_idx,:].cpu().numpy() ,pre2.cpu().numpy())
        mse_2,r2_2 = Evaluator.calculate_metrics(self.data['Y'][train_idx,:].cpu().numpy() ,pre3.cpu().numpy())
        mse_3,r2_3 = Evaluator.calculate_metrics(self.data['M'][train_idx,:].cpu().numpy() ,pre1.cpu().numpy())
        if mse_1+mse_2+mse_3 < self.mse_train_min:
            self.mse_train_min = mse_1+mse_2+mse_3
        if r2_1+r2_2+r2_3 > self.r2_train_max:
            self.r2_train_max = r2_1+r2_2+r2_3
        self.mse_train[self.ind_i,fold+5]=round(mse_1+mse_2+mse_3,4)
        self.r2_train[self.ind_i,fold+5]=round(r2_1+r2_2+r2_3,4)
        
        #Validation performance
        pre1=torch.matmul(self.data['genotype_data'][val_idx,:],self.model.W1.t())
        pre2=torch.matmul(self.data['genotype_data'][val_idx,:],self.model.beta.t())
        pre3=torch.matmul(self.data['genotype_data'][val_idx,:],self.model.beta_.t())+torch.matmul(self.data['M'][val_idx,:],self.model.W2.t())
        mse_1,r2_1 = Evaluator.calculate_metrics(self.data['Y'][val_idx,:].cpu().numpy() ,pre2.cpu().numpy())
        mse_2,r2_2 = Evaluator.calculate_metrics(self.data['Y'][val_idx,:].cpu().numpy() ,pre3.cpu().numpy())
        mse_3,r2_3 = Evaluator.calculate_metrics(self.data['M'][val_idx,:].cpu().numpy() ,pre1.cpu().numpy())
        if mse_1+mse_2+mse_3 < self.mse_val_min:
            self.mse_val_min = mse_1+mse_2+mse_3
        if r2_1+r2_2+r2_3 > self.r2_val_max:
            self.r2_val_max = r2_1+r2_2+r2_3
        self.mse_val[self.ind_i,fold+5]=round(mse_1+mse_2+mse_3,4)
        self.r2_val[self.ind_i,fold+5]=round(r2_1+r2_2+r2_3,4)
        
        self.mse_val[:,10]=np.mean(self.mse_val[:,5:9],axis=1)
        self.r2_val[:,10]=np.mean(self.r2_val[:,5:9],axis=1)

    
    def mkdir(self,path):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path) 

    def _save_metrics(self):
        output_dir=f'{self.cfg.result_dir}/{self.cfg.image}/lamda_1_{self.lamda_1}/lamda_2_{self.lamda_2}/lamda_3_{self.lamda_3}/lamda_4_{self.lamda_4}/rho_{self.rho}'
        self.mkdir(output_dir)      
        Evaluator.save_results(self.model.W1.cpu().numpy(), f'{output_dir}/W1.txt')
        Evaluator.save_results(self.model.W2.cpu().numpy(), f'{output_dir}/W2.txt')
        Evaluator.save_results(self.model.beta.cpu().numpy(), f'{output_dir}/beta.txt')
        Evaluator.save_results(self.model.beta_.cpu().numpy(), f'{output_dir}/beta_.txt')
    def _update_W2(self, train_idx):
        print('update W2')
        start_col_p=0
        end_col_p=self.n_features
        M_batch=self.data['M'][train_idx,start_col_p:end_col_p]
        d_w_max=torch.zeros(1,device=self.cfg.device)
        w_max=torch.zeros(1,device=self.cfg.device)

        for f_iter in range(end_col_p-start_col_p):  # Loop over coordinates
            ii = f_iter
            
            
            w_ii = self.model.W2[:, ii+start_col_p].clone() # Store previous value
            self.R2 = self.R2+ torch.outer(M_batch[:, ii].to(torch.float32), w_ii) # rank 1 update

            tmp = ((self.loss_w2*torch.matmul(M_batch[:, ii].to(torch.float32).t(),self.R2)+self.rho*(torch.matmul(self.model.W1[ii,:],(self.model.beta-self.model.beta_).t())))/(self.loss_w2*torch.matmul(self.data['M'][:,ii].t(),self.data['M'][:,ii]) + self.rho*torch.matmul(self.model.W1[ii,:],self.model.W1[ii,:].t()))).view(-1)
            
            self.model.W2[:, ii+start_col_p] = torch.clamp((torch.abs(tmp)-self.lamda_2),min=0)*torch.sign(tmp)
            W2_gpu=self.model.W2[:, ii+start_col_p].to(torch.float16)
            self.R2 = self.R2-torch.outer(M_batch[:, ii], W2_gpu)
            d_w_ii = diff_abs_max(W2_gpu, w_ii[:])
            if d_w_ii > d_w_max:
                d_w_max = d_w_ii
            W_ii_abs_max = abs_max(W2_gpu)
            if W_ii_abs_max > w_max:
                w_max = W_ii_abs_max
        self.w2_max=w_max
        self.d_w2_max=d_w_max

    
    def _update_W1(self, train_idx):
        print('update W1')
        d_w_max=torch.zeros(1,device=self.cfg.device)
        w_max=torch.zeros(1,device=self.cfg.device)
        for th in range(self.n_batch):
    
            start_col=0+th*self.cfg.chunk_size
            end_col=min(0+(th+1)*self.cfg.chunk_size,self.num_snps)
            snp_chunk_filled=self.data['genotype_data'][train_idx,start_col:end_col]
            for f_iter in range(end_col-start_col):  # Loop over coordinates
                ii = f_iter
                mu=torch.inverse(torch.diag(self.loss_w3).to(torch.float32)*torch.matmul(snp_chunk_filled[:,ii].t(),snp_chunk_filled[:,ii])+self.rho*torch.matmul(self.model.W2.t(),self.model.W2))
                w_ii = self.model.W1[:, ii+start_col].clone() # Store previous value
                self.R3 = self.R3+torch.outer(snp_chunk_filled[:, ii].to(torch.float32), w_ii) # rank 1 update
                tmp = torch.matmul((torch.mul(self.loss_w3,torch.matmul(snp_chunk_filled[:, ii].to(torch.float32).t(),self.R3))+self.rho*torch.matmul((self.model.beta[:,ii+start_col]-self.model.beta_[:,ii+start_col]).t(),self.model.W2)),mu).view(-1)
                self.model.W1[:, ii+start_col] = torch.clamp((torch.abs(tmp)-self.lamda_1),min=0)*torch.sign(tmp)
                W1_gpu=self.model.W1[:, ii+start_col].to(torch.float16)
                self.R3 = self.R3-torch.outer(snp_chunk_filled[:, ii], W1_gpu).to(torch.float32)
        
                d_w_ii = diff_abs_max(W1_gpu, w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii
                W_ii_abs_max = abs_max(W1_gpu)
                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max

        self.w1_max=w_max

        self.d_w1_max=d_w_max

    def _update_beta_(self,train_idx):

        print('update beta_')
        d_w_max=torch.zeros(1,device=self.cfg.device)
        w_max=torch.zeros(1,device=self.cfg.device)
        for th in range(self.n_batch):
            start_col=0+th*self.cfg.chunk_size
            end_col=min(0+(th+1)*self.cfg.chunk_size,self.num_snps)
            snp_chunk_filled=self.data['genotype_data'][train_idx,start_col:end_col]
            
          
            for f_iter in range(end_col-start_col):  # Loop over coordinates
                ii = f_iter
                
                w_ii = self.model.beta_[:, ii+start_col].clone() # Store previous value
                self.R2 = self.R2+torch.outer(snp_chunk_filled[:, ii].to(torch.float32), w_ii) # rank 1 update

                tmp = ((self.loss_w2*torch.matmul(snp_chunk_filled[:, ii].to(torch.float32).t(),self.R2)+self.rho*(self.model.beta[:,ii+start_col].t()-(torch.matmul(self.model.W2,self.model.W1[:,ii+start_col]).t())))/(self.loss_w2*torch.matmul(snp_chunk_filled[:,ii].t(),snp_chunk_filled[:,ii]) + self.rho)).view(-1)
  
                self.model.beta_[:, ii+start_col] = torch.clamp((torch.abs(tmp)-self.lamda_4),min=0)*torch.sign(tmp)
                beta_gpu=self.model.beta_[:, ii+start_col].to(torch.float16)
                self.R2 = self.R2-torch.outer(snp_chunk_filled[:, ii], beta_gpu).to(torch.float32)

             
                d_w_ii = diff_abs_max(beta_gpu, w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii
                W_ii_abs_max = abs_max(beta_gpu)
                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max
 
        self._beta_max=w_max

        self._d_beta_max=d_w_max

    
    def _update_beta(self, train_idx):

        print('update beta')
        d_w_max=torch.zeros(1,device=self.cfg.device)
        w_max=torch.zeros(1,device=self.cfg.device)
        for th in range(self.n_batch):
            start_col=0+th*self.cfg.chunk_size
            end_col=min(0+(th+1)*self.cfg.chunk_size,self.num_snps)
            snp_chunk_filled=self.data['genotype_data'][train_idx,start_col:end_col]
           
            for f_iter in range(end_col-start_col):  # Loop over coordinates
                ii = f_iter
               
                w_ii = self.model.beta[:, ii+start_col].clone() # Store previous value
                self.R1 = self.R1+torch.outer(snp_chunk_filled[:, ii].to(torch.float32), w_ii) # rank 1 update

                tmp = ((self.loss_w1*torch.matmul(snp_chunk_filled[:, ii].to(torch.float32).t(),self.R1)+self.rho*(self.model.beta_[:,ii+start_col].t()+(torch.matmul(self.model.W2,self.model.W1[:,ii+start_col])).t()))/(self.loss_w1*torch.matmul(snp_chunk_filled[:,ii].t(),snp_chunk_filled[:,ii]) + self.rho)).view(-1)
                self.model.beta[:, ii+start_col] = torch.clamp((torch.abs(tmp)-self.lamda_3),min=0)*torch.sign(tmp)
                beta_gpu=self.model.beta[:, ii+start_col].to(torch.float16)
                self.R1 = self.R1-torch.outer(snp_chunk_filled[:, ii], beta_gpu).to(torch.float32)
                      
                d_w_ii = diff_abs_max(beta_gpu, w_ii)
                if d_w_ii > d_w_max:
                    d_w_max = d_w_ii
                W_ii_abs_max = abs_max(beta_gpu)
                if W_ii_abs_max > w_max:
                    w_max = W_ii_abs_max
  
        self.beta_max=w_max

        self.d_beta_max=d_w_max

