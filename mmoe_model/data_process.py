import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import pandas as pd
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


random.seed(3)
np.random.seed(3)
seed = 3
batch_size = 1024

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1]==1 and len(input_shape)>1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()  # 拉成一维矩阵
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)
    return categorical


def data_preparation():
    column_names = ['country_y', 'yearmonth', 'media_gained_decisiondate', 
            'media_total_decisiondate', 'followers_gained_decisiondate', 'followers_total_decisiondate', 'following_gained_decisiondate', 
            'following_total_decisiondate', 'media_gained_decisiondate_7days', 'media_total_decisiondate_7days', 'followers_gained_decisiondate_7days', 
            'followers_total_decisiondate_7days', 'following_gained_decisiondate_7days', 'following_total_decisiondate_7days', 'media_gained_decisiondate_15days', 
            'media_total_decisiondate_15days', 'followers_gained_decisiondate_15days', 'followers_total_decisiondate_15days', 'following_gained_decisiondate_15days', 
            'following_total_decisiondate_15days', 'media_gained_decisiondate_30days', 'media_total_decisiondate_30days', 'followers_gained_decisiondate_30days', 
            'followers_total_decisiondate_30days', 'following_gained_decisiondate_30days', 'following_total_decisiondate_30days', 'media_gained_decisiondate_60days', 
            'media_total_decisiondate_60days', 'followers_gained_decisiondate_60days', 'followers_total_decisiondate_60days', 'following_gained_decisiondate_60days', 
            'following_total_decisiondate_60days', 'media_gained_decisiondate_90days', 'media_total_decisiondate_90days', 'followers_gained_decisiondate_90days', 
            'followers_total_decisiondate_90days', 'following_gained_decisiondate_90days', 'following_total_decisiondate_90days', 'followers_rate_decisiondate_7days', 
            'following_rate_decisiondate_7days', 'media_rate_decisiondate_7days', 'followers_rate_decisiondate_15days', 'following_rate_decisiondate_15days', 
            'media_rate_decisiondate_15days', 'followers_rate_decisiondate_30days', 'following_rate_decisiondate_30days', 'media_rate_decisiondate_30days', 
            'followers_rate_decisiondate_60days', 'following_rate_decisiondate_60days', 'media_rate_decisiondate_60days', 'followers_rate_decisiondate_90days', 
            'following_rate_decisiondate_90days', 'media_rate_decisiondate_90days',  'avg_total_orders', 'avg_new_customers', 'avg_customer_num', 
            'first_campaign_time', 'new_influencer', 'last_orders_total', 'last_new_customers', 'last_customer_num', 'days_from_last_posting', 
            'days_from_last_sponsored', 'days_from_last_org_branded', 'days_from_last_org_nonbranded', 'days_interval_posting', 'days_interval_sponsored', 
            'days_interval_org_branded', 'days_interval_org_nonbranded', '90days_comment_count', '90days_like_count', 'num_sponsored_posts_90days', 
            'num_organic_branded_90days', 'num_organic_nonbranded_90days', 'num_posts_90days', '60days_comment_count', '60days_like_count', 
            'num_sponsored_posts_60days', 'num_organic_branded_60days', 'num_organic_nonbranded_60days', 'num_posts_60days', '30days_comment_count', 
            '30days_like_count', 'num_sponsored_posts_30days', 'num_organic_branded_30days', 'num_organic_nonbranded_30days', 'num_posts_30days', 
            '15days_comment_count', '15days_like_count', 'num_sponsored_posts_15days', 'num_organic_branded_15days', 'num_organic_nonbranded_15days', 
            'num_posts_15days', '7days_comment_count', '7days_like_count', 'num_sponsored_posts_7days', 'num_organic_branded_7days', 'num_organic_nonbranded_7days', 
            'num_posts_7days', 'accept', 'Revenue', 'reputation_change_v2']

    df = pd.read_csv('/var/tmp/code/recommendation_model/mmoe_model/data/precampaign_duringcampaign_features_acceptance_decision_performance_0502.csv', delimiter=',', index_col=None, low_memory=False)
    df = df[column_names]

#     train_df = pd.read_csv('/var/tmp/code/recommendation_model/mmoe_model/data/census-income.data.gz', delimiter=',', header=None, index_col=None, names=column_names)
#     test_df = pd.read_csv('/var/tmp/code/recommendation_model/mmoe_model/data/census-income.test.gz', delimiter=',', header=None, index_col=None, names=column_names)

    # 论文中第一组任务
    label_columns = ['accept', 'Revenue', 'reputation_change_v2']

    categorical_columns = ['first_campaign_time', 'country_y', 'yearmonth']

    df.fillna(0, inplace=True)
    df_transformed = pd.get_dummies(df.drop(label_columns, axis=1), columns=categorical_columns)
    df_labels = df[label_columns]


    return df_transformed, df_labels


def getTensorDataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y)
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)


df_transformed, df_labels = data_preparation()
df_labels_np = df_labels.to_numpy()

# df_label_tmp = np.column_stack((np.argmax(df_labels_np_tp[0]), np.argmax(df_labels_np_tp[1]), np.argmax(df_labels_np_tp[2])))
df_dataset = getTensorDataset(df_transformed.to_numpy(), df_labels_np)
train_size = int(0.8 * len(df_dataset))
test_size = len(df_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset=df_dataset, lengths=[train_size, test_size], generator=torch.manual_seed(0))

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
