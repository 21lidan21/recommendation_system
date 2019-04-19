# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:18:57 2019

@author: 
训练日前1日/前5日两个时间段，
转化率特征：
算出用户\商品\店铺\品牌\类别\类目\上下文ID\检索词预测品类\预测一级类目\预测类目的转化率，之后算出从商品-预测类目两两交叉的转化率；

用户\商品\店铺\品牌\类别的转化率在一些特征下的排名，具体为：
用户的转化率在店铺\品牌\类别\城市的排名；
商品的转化率在店铺\品牌\类别\城市\一级类目\类目下的排名；
店铺的转化率在店铺\品牌\类别\一级类目\类目下的排名；
品牌的转化率在店铺\城市\一级类目\类目下的排名；
类别转化率在城市\店铺下的排名

用户的“下单数&点击数”组合字段

"""
import pandas as pd   

"""
23号之前所有天的统计特征
用户/商品/品牌/店铺/类别/城市/page/query 转化率(buy/cnt+3)

""" 
def all_days_feature(org,trainday,writefilename): 
    #准备列
    col=['user_id','item_id','item_brand_id','shop_id','item_category_list','item_city_id','query1','query','context_page_id','predict_category_property']
    items=col[1:]
    #从预测日之前的数据里计算出每个人的转化率
    data=org[(org['day']<trainday) & (org['day']>=trainday-5)]  
    user=data.groupby('user_id',as_index=False)['is_trade'].agg({'user_buy':'sum','user_cnt':'count'})
    user['user_7days_cvr']=(user['user_buy'])/(user['user_cnt']+3) 
    #把用户的转化率加到train表里
    train=org[org['day']==trainday][['instance_id']+col]
    train=pd.merge(train,user[['user_id','user_7days_cvr']],on='user_id',how='left')
    #继用户ID之后把其余所有的维度都算一下转化率并且加到表里
    for item in items:
        tmp=data.groupby(item,as_index=False)['is_trade'].agg({item+'_buy':'sum',item+'_cnt':'count'})
        tmp[item+'_7days_cvr'] = tmp[item+'_buy'] / tmp[item+'_cnt']
        train = pd.merge(train, tmp[[item, item+'_7days_cvr']], on=item, how='left')
        print(item)
    #计算除了用户ID以外其他ID的两两交叉的转化率
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            tmp = data.groupby(egg, as_index=False)['is_trade'].agg({'_'.join(egg) + '_buy': 'sum', '_'.join(egg) + '_cnt': 'count'})
            tmp['_'.join(egg) + '_7days_cvr'] = tmp['_'.join(egg) + '_buy'] / tmp['_'.join(egg) + '_cnt']
            train = pd.merge(train, tmp[egg+['_'.join(egg) + '_7days_cvr']], on=egg, how='left')
            print(egg)
    train.drop(col, axis=1).to_csv('../data/'+writefilename,index=False)
    return train

"""
用户行为（购买数、点击数）在昨天和前几天的数值
"""
def user_encoder_feature(org,trainday,writefilename):
    train = org[org['day'] == trainday]
    #看前5天的转化率
    data = org[(org['day']<trainday) & (org['day']>=trainday-5)]
    user7 = data.groupby('user_id', as_index=False)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
    user7['user_allday_buy_click']=user7.apply(lambda x:str(x['user_buy'])+'-'+str(x['user_cnt']),axis=1)
    #看昨天的转化率
    data=org[org['day'] == trainday-1]
    user6=data.groupby('user_id', as_index=False)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
    user6['user_6day_buy_click'] = user6.apply(lambda x: str(x['user_buy']) + '-' + str(x['user_cnt']), axis=1)
    train = pd.merge(train, user7, on='user_id', how='left')
    train = pd.merge(train, user6, on='user_id', how='left')
    train[['instance_id','user_allday_buy_click','user_6day_buy_click']].to_csv('../data/'+writefilename)

"""
23号前一天，22号的统计特征
用户/商品/品牌/店铺/类别/城市 转化率 

"""
def latest_day_feature(org,trainday,writefilename):
    data = org[org['day'] ==trainday-1]
    col = ['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id', 'query1', 'query','context_page_id','predict_category_property']
    train = org[org['day'] == trainday][['instance_id'] + col]
    user = data.groupby('user_id', as_index=False)['is_trade'].agg({'user_buy': 'sum', 'user_cnt': 'count'})
    user['user_6day_cvr'] = (user['user_buy']) / (user['user_cnt'] + 3)
    train = pd.merge(train, user[['user_id', 'user_6day_cvr']], on='user_id', how='left')
    items = col[1:]
    for item in items:
        tmp=data.groupby(item,as_index=False)['is_trade'].agg({item+'_buy':'sum',item+'_cnt':'count'})
        tmp[item+'_6day_cvr'] = tmp[item+'_buy'] / tmp[item+'_cnt']
        train = pd.merge(train, tmp[[item, item+'_6day_cvr']], on=item, how='left')
        print(item)
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            tmp = data.groupby(egg, as_index=False)['is_trade'].agg({'_'.join(egg) + '_buy': 'sum', '_'.join(egg) + '_cnt': 'count'})
            tmp['_'.join(egg) + '_6day_cvr'] = tmp['_'.join(egg) + '_buy'] / tmp['_'.join(egg) + '_cnt']
            train = pd.merge(train, tmp[egg+['_'.join(egg) + '_6day_cvr']], on=egg, how='left')
            print(egg)
    train.drop(col, axis=1).to_csv('../data/'+writefilename,index=False)
    return train
 
"""
中间函数，指定两个时间段，用一个时间段的转化率做为另一个时间段数据的特征
"""

"""
#todo
排名特征
前7天的算一次，第7天的算一次
用户转化率在品牌，店铺，类别，城市下面的排名

商品转化率在店铺下面的排名
商品转化率在品牌下面的排名
商品转化率在类别下面的排名
商品转化率在城市下面的排名
商品转化率在query1下面的排名
商品转化率在query下面的排名

店铺转化率在品牌下面的排名
店铺转化率在城市下面的排名
店铺转化率在类别下面的排名
店铺转化率在query1下面的排名
店铺转化率在query下面的排名

品牌在城市下面的转化率排名
品牌在店铺下面的转化率排名
品牌转化率在query1下面的排名
品牌转化率在query下面的排名

类别在城市下面的转换率排名
类别在店铺下面的转换率排名
"""
def rank_6day_feature(data,writefilename):
    data['user_cvr_brand_6day_rank']=data.groupby('item_brand_id')['user_6day_cvr'].rank(ascending=False,method='dense')
    data['user_cvr_shop_6day_rank'] = data.groupby('shop_id')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_cate_6day_rank'] = data.groupby('item_category_list')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_city_6day_rank'] = data.groupby('item_city_id')['user_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_brand_6day_rank'] = data.groupby('item_brand_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_cate_6day_rank'] = data.groupby('item_category_list')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_brand_6day_rank'] = data.groupby('item_brand_id')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_cate_6day_rank'] = data.groupby('item_category_list')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_city_6day_rank'] = data.groupby('item_city_id')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_city_6day_rank'] = data.groupby('item_city_id')['item_category_list_6day_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_shop_6day_rank'] = data.groupby('shop_id')['item_category_list_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query_6day_rank'] = data.groupby('query')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query1_6day_rank'] = data.groupby('query1')['item_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query_6day_rank'] = data.groupby('query')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query1_6day_rank'] = data.groupby('query1')['shop_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query_6day_rank'] = data.groupby('query')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query1_6day_rank'] = data.groupby('query1')['item_brand_id_6day_cvr'].rank(ascending=False, method='dense')
    data=data[['instance_id','user_cvr_brand_6day_rank','user_cvr_shop_6day_rank','user_cvr_cate_6day_rank','user_cvr_city_6day_rank','item_cvr_shop_6day_rank','item_cvr_brand_6day_rank','item_cvr_cate_6day_rank','item_cvr_city_6day_rank','shop_cvr_brand_6day_rank','shop_cvr_cate_6day_rank','shop_cvr_city_6day_rank','brand_cvr_city_6day_rank','brand_cvr_shop_6day_rank','cate_cvr_city_6day_rank','cate_cvr_shop_6day_rank','item_cvr_query_6day_rank','item_cvr_query1_6day_rank','shop_cvr_query_6day_rank','shop_cvr_query1_6day_rank','brand_cvr_query_6day_rank','brand_cvr_query1_6day_rank'
    ]]
    data.to_csv('../data/'+writefilename,index=False)

def rank_7days_feature(data,writefilename):
    data['user_cvr_brand_7days_rank']=data.groupby('item_brand_id')['user_7days_cvr'].rank(ascending=False,method='dense')
    data['user_cvr_shop_7days_rank'] = data.groupby('shop_id')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_cate_7days_rank'] = data.groupby('item_category_list')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['user_cvr_city_7days_rank'] = data.groupby('item_city_id')['user_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_brand_7days_rank'] = data.groupby('item_brand_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_cate_7days_rank'] = data.groupby('item_category_list')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_brand_7days_rank'] = data.groupby('item_brand_id')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_cate_7days_rank'] = data.groupby('item_category_list')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_city_7days_rank'] = data.groupby('item_city_id')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_city_7days_rank'] = data.groupby('item_city_id')['item_category_list_7days_cvr'].rank(ascending=False, method='dense')
    data['cate_cvr_shop_7days_rank'] = data.groupby('shop_id')['item_category_list_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query_7days_rank'] = data.groupby('query')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['item_cvr_query1_7days_rank'] = data.groupby('query1')['item_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query_7days_rank'] = data.groupby('query')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['shop_cvr_query1_7days_rank'] = data.groupby('query1')['shop_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query_7days_rank'] = data.groupby('query')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data['brand_cvr_query1_7days_rank'] = data.groupby('query1')['item_brand_id_7days_cvr'].rank(ascending=False, method='dense')
    data=data[['instance_id','user_cvr_brand_7days_rank','user_cvr_shop_7days_rank','user_cvr_cate_7days_rank','user_cvr_city_7days_rank','item_cvr_shop_7days_rank','item_cvr_brand_7days_rank','item_cvr_cate_7days_rank','item_cvr_city_7days_rank','shop_cvr_brand_7days_rank','shop_cvr_cate_7days_rank','shop_cvr_city_7days_rank','brand_cvr_city_7days_rank','brand_cvr_shop_7days_rank','cate_cvr_city_7days_rank','cate_cvr_shop_7days_rank','item_cvr_query_7days_rank','item_cvr_query1_7days_rank','shop_cvr_query_7days_rank','shop_cvr_query1_7days_rank','brand_cvr_query_7days_rank','brand_cvr_query1_7days_rank'
    ]]
    data.to_csv('../data/'+writefilename,index=False)

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    trainday = 23
    user_encoder_feature(org,trainday,'user_buy_click_feature_train.csv')
    rank_7days_feature(all_days_feature(org,trainday,'7days_cvr_feature_train.csv'),'rank_feature_7days_train.csv')
    rank_6day_feature(latest_day_feature(org,trainday,'6day_cvr_feature_train.csv'),'rank_feature_6day_train.csv')
    trainday = 24
    user_encoder_feature(org,trainday,'user_buy_click_feature_test.csv')
    rank_7days_feature(all_days_feature(org,trainday,'7days_cvr_feature_test.csv'),'rank_feature_7days_test.csv')
    rank_6day_feature(latest_day_feature(org,trainday,'6day_cvr_feature_test.csv'),'rank_feature_6day_test.csv') 

