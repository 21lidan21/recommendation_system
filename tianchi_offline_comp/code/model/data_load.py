# -*- coding: utf-8 -*

#################### file path ####################

path_df_part_1_uic_label_cluster = "././data/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "././data/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic = "././data/mobile/df_part_3_uic.csv"

# data_set features
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

import pandas as pd
import numpy as np

import gc

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

## shuffle (important for keep data distribution) 
# 最简单的方法就是采用pandas中自带的 sample这个方法。
# 假设df是这个DataFrame
# df.sample(frac=1)
# 这样对可以对df进行shuffle。其中参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3。
# 有时候，我们可能需要打混后数据集的index（索引）还是按照正常的排序。我们只需要这样操作

# df.sample(frac=1).reset_index(drop=True)
# 打散 抽取10%比例 index（索引）还是按照正常的排序
df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.sample(frac=1, random_state=1).reset_index(drop=True) 
df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.sample(frac=1, random_state=1).reset_index(drop=True) 
print('========================================')
print(df_part_1_uic_label_cluster.columns)
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

##### generation and splitting to training set & valid set for cross-validation
def valid_train_set_split(folds, fold, train_np_ratio=1, valid_sub_ratio=1, train_sub_ratio=1, seed=None):

    df_part_1_uic_label_cluster_pos = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class']==0]
    df_part_1_uic_label_cluster_neg = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class']!=0]
    df_part_2_uic_label_cluster_pos = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class']==0]
    df_part_2_uic_label_cluster_neg = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class']!=0]

    # generation of mask
    n1_pos = int(len(df_part_1_uic_label_cluster_pos)/folds - 1)
    n1_neg = int(len(df_part_1_uic_label_cluster_neg)/folds - 1)    
    n2_pos = int(len(df_part_2_uic_label_cluster_pos)/folds - 1)
    n2_neg = int(len(df_part_2_uic_label_cluster_neg)/folds - 1)     
    print("***********************")
    print(n1_pos)
    print("***********************")
    msk1_pos = np.zeros(len(df_part_1_uic_label_cluster_pos), dtype=int)
    msk1_neg = np.zeros(len(df_part_1_uic_label_cluster_neg), dtype=int)    
    msk2_pos = np.zeros(len(df_part_2_uic_label_cluster_pos), dtype=int)
    msk2_neg = np.zeros(len(df_part_2_uic_label_cluster_neg), dtype=int)    

    msk1_pos[n1_pos * fold : n1_pos * fold + n1_pos] = 1   
    msk1_neg[n1_neg * fold : n1_neg * fold + n1_neg] = 1       
    msk2_pos[n2_pos * fold : n2_pos * fold + n2_pos] = 1   
    msk2_neg[n2_neg * fold : n2_neg * fold + n2_neg] = 1        

    ###### get the valid set indices [u-i-c-label-class]
    valid_df_part_1_index     = df_part_1_uic_label_cluster_pos[msk1_pos==1].sample(frac = valid_sub_ratio, random_state=seed)
    valid_df_part_1_index_neg = df_part_1_uic_label_cluster_neg[msk1_neg==1]   
    valid_df_part_2_index     = df_part_2_uic_label_cluster_pos[msk2_pos==1].sample(frac = valid_sub_ratio, random_state=seed)
    valid_df_part_2_index_neg = df_part_2_uic_label_cluster_neg[msk2_neg==1]     
    
    for i in range(1,1001,1):
        valid_df_part_1_index_neg_i = valid_df_part_1_index_neg[valid_df_part_1_index_neg['class'] == i]
        if len(valid_df_part_1_index_neg_i) != 0:
            valid_df_part_1_index_neg_i = valid_df_part_1_index_neg_i.sample(frac = valid_sub_ratio, random_state=seed)
            valid_df_part_1_index = pd.concat([valid_df_part_1_index, valid_df_part_1_index_neg_i])
        
        valid_df_part_2_index_neg_i = valid_df_part_2_index_neg[valid_df_part_2_index_neg['class'] == i]
        if len(valid_df_part_2_index_neg_i) != 0:
            valid_df_part_2_index_neg_i = valid_df_part_2_index_neg_i.sample(frac = valid_sub_ratio, random_state=seed)
            valid_df_part_2_index = pd.concat([valid_df_part_2_index, valid_df_part_2_index_neg_i])

    ###### get the train set indices [u-i-c-label-class]
    train_df_part_1_index     = df_part_1_uic_label_cluster_pos[msk1_pos==0].sample(frac = train_sub_ratio, random_state=seed)
    train_df_part_1_index_neg = df_part_1_uic_label_cluster_neg[msk1_neg==0]  
    train_df_part_2_index     = df_part_2_uic_label_cluster_pos[msk2_pos==0].sample(frac = train_sub_ratio, random_state=seed)
    train_df_part_2_index_neg = df_part_2_uic_label_cluster_neg[msk2_neg==0]     
    
    frac_ratio = train_sub_ratio * train_np_ratio/1200  
    
    for i in range(1,1001,1):
        train_df_part_1_index_neg_i = train_df_part_1_index_neg[train_df_part_1_index_neg['class'] == i]
        if len(train_df_part_1_index_neg_i) != 0:
            train_df_part_1_index_neg_i = train_df_part_1_index_neg_i.sample(frac = frac_ratio, random_state=seed)
            train_df_part_1_index = pd.concat([train_df_part_1_index, train_df_part_1_index_neg_i])
        
        train_df_part_2_index_neg_i = train_df_part_2_index_neg[train_df_part_2_index_neg['class'] == i]
        if len(train_df_part_2_index_neg_i) != 0:
            train_df_part_2_index_neg_i = train_df_part_2_index_neg_i.sample(frac = frac_ratio, random_state=seed)
            train_df_part_2_index = pd.concat([train_df_part_2_index, train_df_part_2_index_neg_i])

    ###### constructing valid set 
    valid_part_1_df = pd.merge(valid_df_part_1_index, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_df_part_2_index, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])
    
    del(valid_part_1_df)
    del(valid_part_2_df)
    del(valid_df_part_1_index)
    del(valid_df_part_1_index_neg)
    del(valid_df_part_2_index)
    del(valid_df_part_2_index_neg)
        
    gc.collect()  
    
    valid_df.fillna(-1, inplace=True)
    print("valid subset is generated.")
    
    train_part_1_df = pd.merge(train_df_part_1_index, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_df_part_2_index, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])
    
    del(train_part_1_df)
    del(train_part_2_df)
    del(train_df_part_1_index)
    del(train_df_part_1_index_neg)
    del(train_df_part_2_index)
    del(train_df_part_2_index_neg)
    
    gc.collect()   
    
    train_df.fillna(-1, inplace=True)
    print("train subset is generated.")
    return valid_df, train_df


def valid_train_set_construct(valid_ratio = 0.5, valid_sub_ratio = 0.5, train_np_ratio = 1, train_sub_ratio = 0.5, seed=None):
    rng = np.random.RandomState(seed)
    msk_1 = rng.rand(len(df_part_1_uic_label_cluster)) < valid_ratio
    msk_2 = rng.rand(len(df_part_2_uic_label_cluster)) < valid_ratio
        
    valid_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[msk_1]
    valid_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[msk_2]
    
    valid_part_1_uic_label = valid_df_part_1_uic_label_cluster[ valid_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio, random_state=seed)
    valid_part_2_uic_label = valid_df_part_2_uic_label_cluster[ valid_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = valid_sub_ratio, random_state=seed)
    
    ### constructing valid set
    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = valid_df_part_1_uic_label_cluster[valid_df_part_1_uic_label_cluster['class'] == i]
        if len(valid_part_1_uic_label_0_i) != 0:
            valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = valid_sub_ratio, random_state=seed)
            valid_part_1_uic_label     = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
        
        valid_part_2_uic_label_0_i = valid_df_part_2_uic_label_cluster[valid_df_part_2_uic_label_cluster['class'] == i]
        if len(valid_part_2_uic_label_0_i) != 0:
            valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = valid_sub_ratio, random_state=seed)
            valid_part_2_uic_label     = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
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
    
    valid_df = pd.concat([valid_part_1_df, valid_part_2_df])
    # fill the missing value as -1 (missing value are time features)
    valid_df.fillna(-1, inplace=True)
    print("valid subset is generated.")
    
    ### constructing training set
    train_df_part_1_uic_label_cluster = df_part_1_uic_label_cluster.loc[~msk_1]
    train_df_part_2_uic_label_cluster = df_part_2_uic_label_cluster.loc[~msk_2] 
    
    train_part_1_uic_label = train_df_part_1_uic_label_cluster[ train_df_part_1_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio, random_state=seed)
    train_part_2_uic_label = train_df_part_2_uic_label_cluster[ train_df_part_2_uic_label_cluster['class'] == 0 ].sample(frac = train_sub_ratio, random_state=seed)
    
    frac_ratio = train_sub_ratio * train_np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = train_df_part_1_uic_label_cluster[train_df_part_1_uic_label_cluster['class'] == i]
        if len(train_part_1_uic_label_0_i) != 0:
            train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio, random_state=seed)
            train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = train_df_part_2_uic_label_cluster[train_df_part_2_uic_label_cluster['class'] == i]
        if len(train_part_2_uic_label_0_i) != 0:
            train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio, random_state=seed)
            train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    
    # constructing training set
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
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])
    # fill the missing value as -1 (missing value are time features)
    train_df.fillna(-1, inplace=True)
    print("train subset is generated.")
    
    return valid_df, train_df

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
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")
    
    # constructing training set
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
    
    train_df = pd.concat([train_part_1_df, train_part_2_df])

    train_df.fillna(-1, inplace=True)
    
    print("train subset is generated.")
    return train_df

##### generation of data set on data_frame_1
def data_set_construct_by_part(np_ratio = 1, sub_ratio = 1):
     
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    
    frac_ratio = sub_ratio * np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
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

    train_part_1_df.fillna(-1, inplace=True)
    train_part_2_df.fillna(-1, inplace=True)  
      
    print("train subset is generated.")
    return train_part_1_df, train_part_2_df

    

    
    
    
    
    
