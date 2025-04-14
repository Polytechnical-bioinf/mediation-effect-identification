from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

class Evaluator:
    @staticmethod
    def calculate_metrics(Y_true, Y_pred):
        mse = round(mean_squared_error(Y_true, Y_pred), 4)
        r2 = round(r2_score(Y_true, Y_pred), 4)
        return mse, r2
    
    @staticmethod
    def save_results(results, filepath):
        np.savetxt(filepath, results, delimiter=' ', fmt='%.6f')