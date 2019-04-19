# -*- coding: utf-8 -*
    


########## file path ##########
##### input file
path_df_part_1_uic_label_cluster = "././data/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "././data/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic       = "././data/output/df_part_3_uic.csv"

path_df_part_1_U   = "././data/output/feature/df_part_1_U.csv"  
path_df_part_1_I   = "././data/output/feature/df_part_1_I.csv"
path_df_part_1_C   = "././data/output/feature/df_part_1_C.csv"
path_df_part_1_IC  = "././data/output/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "././data/output/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "././data/output/feature/df_part_1_UC.csv"

path_df_part_2_U   = "././data/output/feature/df_part_2_U.csv"  
path_df_part_2_I   = "././data/output/feature/df_part_2_I.csv"
path_df_part_2_C   = "././data/output/feature/df_part_2_C.csv"
path_df_part_2_IC  = "././data/output/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "././data/output/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "././data/output/feature/df_part_2_UC.csv"

path_df_part_3_U   = "././data/output/feature/df_part_3_U.csv"  
path_df_part_3_I   = "././data/output/feature/df_part_3_I.csv"
path_df_part_3_C   = "././data/output/feature/df_part_3_C.csv"
path_df_part_3_IC  = "././data/output/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "././data/output/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "././data/output/feature/df_part_3_UC.csv"

# normalize scaler
path_df_part_1_scaler = "././data/k_means_subsample/df_part_1_scaler"
path_df_part_2_scaler = "././data/k_means_subsample/df_part_2_scaler"

# item_sub_set P
path_df_P = "././data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"

##### output file
path_df_result     = "././data/output/res_LR_k_means_subsample.csv"
path_df_result_tmp = "././data/output/df_result_tmp.csv"

import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt

import time

def df_read(path, mode = 'r'):
   
    data_file = open(path, mode)
    try:     df = pd.read_csv(data_file, index_col = False)
    finally: data_file.close()
    return   df

def subsample(df, sub_size):

    if sub_size >= len(df) : return df
    else : return df.sample(n = sub_size)

##### loading data of part 1 & 2
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U  = df_read(path_df_part_2_U )   
df_part_2_I  = df_read(path_df_part_2_I )
df_part_2_C  = df_read(path_df_part_2_C )
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

# generation of target testing set
part_1_scaler = pickle.load(open(path_df_part_1_scaler,'rb'))
part_2_scaler = pickle.load(open(path_df_part_2_scaler,'rb'))

##### generation of training set & valid set
def train_set_construct(np_ratio = 1, sub_ratio = 1):
    
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    
    frac_ratio = sub_ratio * np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        # print(train_part_2_uic_label_0_i.info)
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        # 把正负样本合并
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    # print(train_part_2_uic_label.head()) 
    print("training subset uic_label keys is selected.")
    
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
       
    train_X_1 = train_part_1_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u'])
    train_y_1 = train_part_1_df['label'].values
    # feature standardization
    standard_train_X_1 = part_1_scaler.transform(train_X_1)

    train_X_2 = train_part_2_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u'])
    train_y_2 = train_part_2_df['label'].values
    standard_train_X_2 = part_2_scaler.transform(train_X_2)
    # concatenate 在行的方向上合并起来
    train_X = np.concatenate((standard_train_X_1, standard_train_X_2))
    train_y = np.concatenate((train_y_1, train_y_2))
    print("train subset is generated.")
    # print(train_X.info)
    # print("888888888888888888888888888888888888")
    # print(train_y.info)
    return train_X, train_y

    
def valid_set_construct(sub_ratio = 0.1):
    
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)

    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
    
        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    # using all the features without missing value for valid lr model
    valid_X_1 = valid_part_1_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u'])
    valid_y_1 = valid_part_1_df['label'].values
    # feature standardization
    standard_valid_X_1 = part_1_scaler.transform(valid_X_1)

    # using all the features without missing value for valid lr model
    valid_X_2 = valid_part_2_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u'])
    valid_y_2 = valid_part_2_df['label'].values
    # feature standardization
    standard_valid_X_2 = part_2_scaler.transform(valid_X_2)
    
    valid_X = np.concatenate((standard_valid_X_1, standard_valid_X_2))
    valid_y = np.concatenate((valid_y_1, valid_y_2))
    print("train subset is generated.")
    
    return valid_X, valid_y

#######################################################################


#######################################################################

##### predicting

df_part_3_U  = df_read(path_df_part_3_U )   
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)
# for get scale transform mechanism to large scale of data
scaler_3 = preprocessing.StandardScaler()
# process by chunk as ui-pairs size is too big
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000): 
    try:     
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])
        
        pred_X = pred_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                    'u_b4_rate',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                    'i_b4_rate',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u'])
        # feature standardization
        scaler_3.partial_fit(pred_X)   
        
        batch += 1
        print("prediction chunk %d done." % batch) 
        
    except StopIteration:
        print("prediction finished.")
        break         


train_X, train_y = train_set_construct(np_ratio=35, sub_ratio=1)

LR_clf = LogisticRegression(verbose=True)
LR_clf.fit(train_X, train_y)

batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000): 
    try:     
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])

        # fill the missing value as -1 (missing value are time features)
        pred_df.fillna(-1, inplace=True)
        
        # using all the features for training RF model
        pred_X = pred_df.as_matrix(['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                    'u_b4_rate',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                    'i_b4_rate',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u'])

        # predicting
        # feature standardization
        standardized_pred_X = scaler_3.transform(pred_X)
        #predict_proba 输出每一类别的概率
        pred_y = (LR_clf.predict_proba(standardized_pred_X)[:,1] > 0.5).astype(int)
        print("pred_y,pred_y,pred_y,pred_y,pred_y")
        # 概率值 >0.5 就是 TRUE / FALSE =》 astype(int) 就转化成0/1
        print(LR_clf.predict_proba(standardized_pred_X)[:,1]> 0.5)
        print("pred_y,pred_y,pred_y,pred_y,pred_y")
        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = pred_y
        # add to result csv
        pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp, 
                                                   columns=['user_id','item_id'],
                                                   index=False, header=False, mode='a')
        
        batch += 1
        print("prediction chunk %d done." % batch) 
        
    except StopIteration:
        print("prediction finished.")
        break         
   
    
#######################################################################
# train-user 原始数据集
df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
print("&&&&&&&&&&&&&&&")
print(df_pred.columns)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)


print(' hhhhhh')

