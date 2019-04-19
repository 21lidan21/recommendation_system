# coding=utf-8
# @author:bryan
"""
top100的特征强制相除交叉
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import lightgbm as lgb 
    
cross_feature_num=100

#计算出前100个重要的特征，和分数
def LGB_test(train_x,train_y,test_x,test_y,cate_col=None):
    #对cate_col的列进行缺失值填充和encoder
    if cate_col:
        data = pd.concat([train_x, test_x])
        for fea in cate_col:
            data[fea]=data[fea].fillna('-1')
            data[fea] = LabelEncoder().fit_transform(data[fea].apply(str))
        train_x=data[:len(train_x)]
        test_x=data[len(train_x):]
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25,random_state=2018,n_jobs=50
    )
    clf.fit(train_x, train_y,eval_set=[(train_x,train_y),(test_x,test_y)],early_stopping_rounds=100)
    feature_importances=sorted(zip(train_x.columns,clf.feature_importances_),key=lambda x:x[1])
    return clf.best_score_[ 'valid_1']['binary_logloss'],feature_importances

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
    score = LGB_test(train_x, train_y, test_x, test_y,cate_col)
    return score[1]

def LGB_predict(data,file):
    data=data.drop(['hour48','hour', 'user_id', 'shop_id','query1','query',
               'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property'], axis=1)
    data['item_category_list'] = LabelEncoder().fit_transform(data['item_category_list'])
    train=data[data['traintype']==1]
    predict=data[data['traintype']==0] 
    res=predict[['instance_id','is_trade']]
    train_y=train.pop('is_trade')
    train_x=train.drop(['day','instance_id'], axis=1)
    test_x = predict.drop(['day', 'instance_id','is_trade'], axis=1)
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=3000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,  # colsample_bylevel=0.7,
        learning_rate=0.01, min_child_weight=25, random_state=2018, n_jobs=50
        ,metric='logloss'
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)])
    res['predicted_score']=clf.predict_proba(test_x)[:,1]
    #testb = pd.read_csv('../data/round2_ijcai_18_test_b_20180510.txt', sep=' ')[['instance_id']]
    #res=pd.merge(testb,res,on='instance_id',how='left')
    #res[['instance_id', 'predicted_score']].to_csv('../submit/' + file + '.txt', sep=' ', index=False)
    logloss = log_loss(res['is_trade'],res['predicted_score'])
    print('final_logloss:',logloss)

if __name__ == '__main__': 
    base_train = pd.read_csv('./data/final_base_train.csv')
    base_train['traintype']=1
    base_test = pd.read_csv('./data/final_base_test.csv')
    base_test['traintype']=0
    #为了避免列不同concat的不便，直接另列明相同
    base_test.columns=base_train.columns
    base = pd.concat([base_train,base_test],axis=0)         
    ##对base_train进行GBDT计算，选出前98个重要的特征  
    features=off_test_split(base)
    feature=[i[0] for i in features[-cross_feature_num:]]
    feature.remove('shop_id') if 'shop_id' in feature else 0 
    feature.remove('item_id') if 'item_id' in feature else 0 
    #对base_train计算98个重要特征的两两的特征， 并再次取前100个    
    cross=base[['hour48', 'hour',  'user_id','query1','query','is_trade','day','item_category_list',
             'instance_id', 'item_property_list', 'context_id', 'context_timestamp', 'predict_category_property','traintype']]
    # shop_id,item_id
    for i in range(len(feature)):
        for j in range(i+1,len(feature)):
            cross['cross_'+str(i)+'_'+str(j)]=base_train[feature[i]]/base_train[feature[j]]
            print('cross_'+str(i)+'_'+str(j))
    #对新生成的TOP100特征的数据进行计算
    score=off_test_split(cross)
    #从交叉后的特征里再次取出前100个重要的特征
    add_feature=[i[0] for i in score[-cross_feature_num:]]
    #共694+100=794个特征     
    base_add=pd.concat([base,cross[add_feature]],axis=1)
    #拿base_add去训练，并且去预测 
    base_add.to_csv('./data/LGBM_final_base.csv',index=False) 
    LGBM_final_base = pd.read_csv('./data/LGBM_final_base.csv')
    LGB_predict(LGBM_final_base,'submit')
