import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import warnings
warnings.filterwarnings("ignore")

with torch.no_grad():
    y1_predict_array = []
    y1_label_array = []

    y2_predict_array = []
    y2_label_array = []

    y3_predict_array = []
    y3_label_array = []
    for batch in test_loader:
        x_batch2, y1_batch, y2_batch, y3_batch = batch
        x_batch2 = x_batch2.to(device=device)
#             accept_output, revenue_output, reputation_output = model(x_batch2)
        accept_output = model(x_batch2)

#             predicted_accept = accept_output.argmax(dim=1)
#             predicted_revenue = revenue_output.argmax(dim=1)
#             predicted_reputation = reputation_output.argmax(dim=1)


        accept_output_cut = [1 if y > 0 else 0 for y in accept_output]
#             revenue_output_cut = [1 if y > 0 else 0 for y in revenue_output]
#             reputation_output_cut = [1 if y > 0 else 0 for y in reputation_output]

        y1_predict_array.extend(accept_output_cut)
        y1_label_array.extend(y1_batch.reshape(1,-1).tolist()[0])

        y2_predict_array.extend(revenue_output)
        y2_label_array.extend(y2_batch.reshape(1,-1).tolist()[0])

        y3_predict_array.extend(reputation_output)
        y3_label_array.extend(y3_batch.reshape(1,-1).tolist()[0])

#         mse_revenue = mean_squared_error(y_true=y2_label_array, y_pred=y2_predict_array)

    accuracy_accept = accuracy_score(y_true=y1_label_array, y_pred=y1_predict_array)
    precision_score_accept = precision_score(y_true=y1_label_array, y_pred=y1_predict_array)
    recall_score_accept = recall_score(y_true=y1_label_array, y_pred=y1_predict_array)
    f1_score_accept = f1_score(y_true=y1_label_array, y_pred=y1_predict_array)
    print('y1 Accuracy, precision, recall, f1 of accept: {:.2f} {:.2f} {:.2f} {:.2f}'.format(accuracy_accept, precision_score_accept, recall_score_accept, f1_score_accept))


    mse_revenue = mean_squared_error(y_true=y2_label_array, y_pred=y2_predict_array)
    r2_revenue = r2_score(y_true=y2_label_array, y_pred=y2_predict_array)
    print('y2 MSE, R2: {:.2f} {:.2f}'.format(mse_revenue, r2_revenue))


    mse_reputation = mean_squared_error(y_true=y3_label_array, y_pred=y3_predict_array)
    r2_reputation = r2_score(y_true=y3_label_array, y_pred=y3_predict_array)
    print('y3 MSE, R2: {:.2f} {:.2f}'.format(mse_reputation, r2_reputation))
    
    
    
    
#         mse_revenue = mean_squared_error(y_true=y2_label_array, y_pred=y2_predict_array)
#         r2_revenue = r2_score(y_true=y2_label_array, y_pred=y2_predict_array)
#         print('y2 MSE, R2: {:.2f} {:.2f}'.format(mse_revenue, r2_revenue))


#         mse_reputation = mean_squared_error(y_true=y3_label_array, y_pred=y3_predict_array)
#         r2_reputation = r2_score(y_true=y3_label_array, y_pred=y3_predict_array)
#         print('y3 MSE, R2: {:.2f} {:.2f}'.format(mse_reputation, r2_reputation))
