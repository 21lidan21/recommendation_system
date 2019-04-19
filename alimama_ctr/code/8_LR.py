import pandas as pd
import numpy as np
from scipy import sparse# 稀疏矩阵
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

cross_feature_num=100
def LR_test(train_x,train_y,test_x,test_y,cate_col=None):
#对cate_col的列进行缺失值填充和encoder
    if cate_col:
        data = pd.concat([train_x, test_x])
        for fea in cate_col:
            data[fea]=data[fea].fillna('-1')
            data[fea] = LabelEncoder().fit_transform(data[fea].apply(str))
        train_x=data[:len(train_x)]
        test_x=data[len(train_x):]
    print("LR test")
    lr = LogisticRegression()
    lr.fit(train_x, train_y)

#拆分测试集训练集
def off_test_split(org,cate_col=None):
    data = org[org.traintype ==1]
    #以下这些字段倍删除了，但是参与了TOP100重要特征
    data = data.drop(
        ['hour48', 'hour',  'user_id','query1','query',
         'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    y = data.pop('is_trade')
    train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.15, random_state=2018)
    train_x.drop('day', axis=1, inplace=True)
    test_x.drop('day', axis=1, inplace=True)
    # score = LGB_test(train_x, train_y, test_x, test_y,cate_col)
    # return score[1]   
    
# model training
def LR_predict(data,file):
    data=data.drop(['hour48','hour', 'user_id', 'shop_id','query1','query',
               'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    train=data[data['traintype']==1]
    predict=data[data['traintype']==0] 
    res=predict[['instance_id','is_trade']]
    train_y=train.pop('is_trade')
    train_x=train.drop(['day','instance_id'], axis=1)
    test_x = predict.drop(['day', 'instance_id','is_trade'], axis=1)
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    
    res['predicted_score']=lr.predict_proba(test_x)[:,1]
    logloss = log_loss(res['is_trade'],res['predicted_score'])
    print('final_logloss:',logloss)



if __name__ == '__main__': 
    feature_final_base = pd.read_csv('./data/LGBM_final_base.csv')
    feature_final_base_small=feature_final_base[0:5000]
    LR_predict(feature_final_base_small,'submit')