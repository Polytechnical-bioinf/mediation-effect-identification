from data_loader import load_data
from model import MediationModel
from mediation import Mediation


data = load_data()
num_snps = data['genotype_data'].shape[1]
n_features = data['M'].shape[1]
n_tasks = data['Y'].shape[1]
# The optimal parameter
lamda_1, lamda_2, lamda_3, lamda_4, rho=0.1,0.1,0.1,0.1,0.1

model = MediationModel(num_snps, n_features, n_tasks)
trainer = Mediation(model, data)
mse_test,r2_test,mse_eq1,mse_eq2,mse_eq3,r2_eq1,r2_eq2,r2_eq3=trainer.test(lamda_1, lamda_2, lamda_3, lamda_4, rho,n_tasks,num_snps,n_features)
print('testing performance')
print('mean performance of three equations')
print('MSE:',mse_test,'\n','R2:',r2_test)
print('Equation1')
print('MSE:',mse_eq1,'\n','R2:',r2_eq1)
print('Equation2')
print('MSE:',mse_eq2,'\n','R2:',r2_eq2)
print('Equation3')
print('MSE:',mse_eq3,'\n','R2:',r2_eq3)