# -*- coding: utf-8 -*-

import pandas as pd 
""" 
连续未购买特征：
在18-22\22号两个时间段里，用户最大连续未购买点击数；
在22号里，用户购买\未购买的商品\店铺数 
"""

"""连续未购买7个特征，线下提升万0.5""" 
def user_continue_nobuy(org,trainday,writefilename):
    data = org[(org['day']<trainday) & (org['day']>=trainday-5)].sort_values(by=['user_id','context_timestamp'])
    train=org[org.day==trainday][['instance_id','user_id']]
    def f(x):
        max_no_buy=0
        res=[]
        for i in x:
            if i==0:
                max_no_buy+=1
                res.append(max_no_buy)
            else:
                max_no_buy=0
        return 0 if len(res)==0 else max(res)
    user_nobuy= data.groupby('user_id',as_index=False)['is_trade'].agg({'user_continue_nobuy_click_cnt':lambda x:f(x)})
    print('user_continue_nobuy_click_cnt finish')
    data=data[data.day==trainday-1].sort_values(by=['user_id','context_timestamp'])
    day6_user_nobuy=data.groupby('user_id', as_index=False)['is_trade'].agg({'day6_user_continue_nobuy_click_cnt': lambda x: f(x)})
    print('day6_user_continue_nobuy_click_cnt finish')
    train=pd.merge(train,user_nobuy,on='user_id',how='left')
    train = pd.merge(train, day6_user_nobuy, on='user_id', how='left')
    data = org[org['day'] ==trainday-1]
    user_buy_items=data[data.is_trade==1].groupby('user_id', as_index=False)['item_id'].agg({'day6_user_buy_items':lambda x:len(set(x))})
    user_nobuy_items=data[data.is_trade==0].groupby('user_id', as_index=False)['item_id'].agg({'day6_user_nobuy_items': lambda x: len(set(x))})
    user_buy_shops = data[data.is_trade==1].groupby('user_id', as_index=False)['shop_id'].agg({'day6_user_buy_shops': lambda x: len(set(x))})
    user_nobuy_shops = data[data.is_trade==0].groupby('user_id', as_index=False)['shop_id'].agg({'day6_user_nobuy_shops': lambda x: len(set(x))})    
    print('day6_user_nobuy finish')
    train = pd.merge(train,user_buy_items,on='user_id',how='left')
    train = pd.merge(train, user_nobuy_items, on='user_id', how='left')
    train = pd.merge(train, user_buy_shops, on='user_id', how='left')
    train = pd.merge(train, user_nobuy_shops, on='user_id', how='left')
    train['day6_user_items_d_shops']=train['day6_user_nobuy_items']/train['day6_user_nobuy_shops']
    train=train.drop('user_id',axis=1)
    train.to_csv('../data/'+writefilename,index=False)
    print('nobuy_feature finish') 

"""
18-22号每日里商品，品牌，店铺，类目，城市，预测类目属性，上下文、预测一级类目、预测类目的点击数\购买数
以及18-21日后一天的点击数\购买数比前一天的比例变化
这些全部转化为特征，放入23日的明细里
！！！！！这里也是要修改的
"""
 
#某维度在某天的购买数和点击数
def trend_f(data, item):
    tmp = data.groupby([item, 'day'], as_index=False)['is_trade'].agg({'buy': 'sum', 'cnt': 'count'})
    features = []
    for key, df in tmp.groupby(item, as_index=False):
        feature = {}
        feature[item] = key
        for index, row in df.iterrows():
            feature[item + 'buy' + str(int(row['day']))] = row['buy']
            feature[item + 'cnt' + str(int(row['day']))] = row['cnt']
        features.append(feature)
    features = pd.DataFrame(features)
    return features

def trend_feature(org,trainday,writefilename):
    data=org[(org['day']<trainday) & (org['day']>=trainday-5)]
    col = ['item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id',
           'predict_category_property', 'context_page_id', 'query1', 'query']
    train=org[org.day==trainday][['instance_id']+col]
    items=col
    for item in items:
        train=pd.merge(train,trend_f(data, item),on=item,how='left')
        print(item+' finish')
    train=train.drop(items,axis=1)
    for item in items:  #print(item)
        for day in range(trainday-5,trainday-1): #对18-21日进行循环操作，计算这些天后一天比这些天的变化率
            train['_'.join([item,str(day+1),'d',str(day),'cnt'])]=train[item + 'cnt' +str(day+1)]/train[item + 'cnt' +str(day)]
            train['_'.join([item, str(day + 1), 'd', str(day), 'buy'])]=train[item + 'buy' +str(day+1)]/train[item + 'buy' +str(day)]
    train=train[[i for i in train.columns if 'cnt22' not in i]]  #把item_brand_idcnt22排除掉
    train.to_csv('../data/'+writefilename,index=False)
    print('trend_feature finish')

'''
一次性购买：每一次点击均促成了购买
对18-22期间\22两个时间段，计算出一次性购买的次数和总购买的次数，和一次性购买率=一次性购买的次数/总购买的次数
'''  

def oneshot(data,item): 
    tmp = data.groupby([item], as_index=False)['is_trade'].agg({item + '_buy': 'sum'})
    shot = data.groupby([item, 'user_id'], as_index=False)['is_trade'].agg({'is_shot': 'mean'})
    shot = shot[shot.is_shot == 1].groupby([item], as_index=False)['is_shot'].agg({item + 'shot_num': 'count'})
    tmp = pd.merge(tmp, shot, on=[item], how='left')
    tmp[item+'_shot_rate'] = tmp[item +'shot_num'] / tmp[item + '_buy']  
    return tmp[[item,item+'_shot_rate']]
 
def day6_shot_feature(org,trainday):
    data=org[org.day==trainday-1]
    items = ['item_id', 'shop_id', 'query', 'query1']
    train = org[org.day == trainday][['instance_id']+items]
    for item in items:
        train = pd.merge(train, oneshot(data, item), on=item, how='left')
    train=train.drop(items,axis=1)
    train.columns=['instance_id','day6_item_shot_rate','day6_shop_shot_rate','day6_query_shot_rate','day6_query1_shot_rate']
    return train

def oneshot_feature(org,trainday,writefilename):
    data=org[(org['day']<trainday) & (org['day']>=trainday-5)]
    items = ['item_id', 'shop_id', 'query', 'query1']
    train = org[org.day == trainday][['instance_id']+items]
    for item in items:
        train=pd.merge(train,oneshot(data, item),on=item,how='left')
        print(item+' finish')
    train = train.drop(items, axis=1)
    print(train.columns) 
    day6=day6_shot_feature(org,trainday)
    print(day6.columns) 
    train = pd.merge(train, day6, on='instance_id', how='left')
    train.to_csv('../data/'+writefilename, index=False)
    print('oneshot_feature finish')


'''
第一次出现到第一次购买的时间间隔
对18-22期间\22两个时间段，计算出商品\店铺\预测类目属性维度下第一次出现到第一次购买的时间间隔
并将这些计算值做为特征放入23日的明细里
''' 
# 商品，店铺，类别，城市，品牌，query  第一次出现到第一次购买的时间间隔
# 前所有天，第七天
def first_ocr(data,item):
    import numpy as np
    import datetime
    def sec_diff(a, b):
        if (a is np.nan) | (b is np.nan):
            return np.nan
        return (datetime.datetime.strptime(str(b), "%Y-%m-%d %H:%M:%S") - datetime.datetime.strptime(str(a),"%Y-%m-%d %H:%M:%S")).seconds
    ocr=data.groupby(item,as_index=False)['context_timestamp'].agg({'min_ocr_time':'min'})
    buy=data[data.is_trade==1].groupby(item,as_index=False)['context_timestamp'].agg({'min_buy_time':'min'})
    data=pd.merge(ocr,buy,on=item,how='left')
    data[item+'_ocr_buy_diff_day6']=data.apply(lambda x:sec_diff(x['min_ocr_time'],x['min_buy_time']),axis=1)
    return data[[item,item+'_ocr_buy_diff_day6']]

# calc data，join data
def today_ocr(c_data, j_data):
    items=['item_id','shop_id','predict_category_property']
    item_shot=first_ocr(c_data, items[0])
    shop_shot=first_ocr(c_data, items[1])
    query_shot=first_ocr(c_data, items[2])
    j_data=pd.merge(j_data,item_shot,on=items[0],how='left')
    j_data = pd.merge(j_data, shop_shot, on=items[1], how='left')
    j_data = pd.merge(j_data, query_shot, on=items[2], how='left')
    j_data= j_data[['instance_id','item_id_ocr_buy_diff','shop_id_ocr_buy_diff','predict_category_property_ocr_buy_diff']]
    j_data.columns=['instance_id','today_item_id_ocr_buy_diff','today_shop_id_ocr_buy_diff','today_predict_category_property_ocr_buy_diff']
    return j_data
 
def first_ocr_feature(org,trainday,writefilename):
    items=['item_id','query','query1']
    data=org[(org['day']<trainday) & (org['day']>=trainday-5)]
    train=org[org.day==trainday][['instance_id']+items]
    for item in items:
        tmp=first_ocr(data, item)
        tmp.columns=[item,item+'_ocr_buy_diff_all_day']
        train=pd.merge(train,tmp,on=item,how='left')
        print(item)
    data=data[data.day==trainday-1]
    for item in items:
        tmp=first_ocr(data, item)
        train=pd.merge(train,tmp,on=item,how='left')
        print(item) 
    train=train.drop(items, axis=1)
    train.to_csv('../data/'+writefilename,index=False)
    print('ocr_feature finish')


"""
item和shop 属性的变化，第6天和前5天属性的均值的差值，第6天和第5天属性的均值的差值
item属性：item_price_level,item_sales_level,item_collected_level,item_pv_level
shop属性：shop_review_num_level,shop_review_positive_rate,shop_star_level,shop_score_service,shop_score_delivery,shop_score_description
线下可以提升1个万分位
"""
def item_shop_var_feature(org,trainday,writefilename):
    import numpy as np
    col=['item_id','shop_id']
    item_cates=['item_price_level','item_sales_level','item_collected_level','item_pv_level']
    shop_cates=['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
    train=org[org.day==trainday][['instance_id']+col+item_cates+shop_cates]
    #23日和前5天均值的差值
    data=org[(org['day']<trainday) & (org['day']>=trainday-5)]
    for cate in item_cates:
        train=pd.merge(train,data.groupby('item_id',as_index=False)[cate].agg({'item_id_'+cate+'_var':np.std,'item_id_'+cate+'_avg':'mean'}),on='item_id',how='left')
        train['_'.join(['diff',cate,'today_d_7days'])] = train[cate] - train['item_id_'+cate+'_avg']
    for cate in shop_cates:
        train=pd.merge(train,data.groupby('shop_id',as_index=False)[cate].agg({'shop_id_'+cate+'_var':np.std,'shop_id_'+cate+'_avg':'mean'}),on='shop_id',how='left')
        train['_'.join(['diff', cate, 'today_d_7days'])] = train[cate] - train['shop_id_' + cate + '_avg'] 
    #23日和22日的差值
    data = org[org.day == trainday-1]
    for cate in item_cates:
        avg=data.groupby('item_id',as_index=False)[cate].agg({'item_id_day6'+cate+'_avg':'mean'})
        tmp=pd.merge(train,avg,on='item_id',how='left')
        train['_'.join(['diff',cate,'today_d_6day'])]=tmp[cate]-tmp['item_id_day6'+cate+'_avg']
    for cate in shop_cates:
        avg=data.groupby('shop_id',as_index=False)[cate].agg({'shop_id_day6'+cate+'_avg':'mean'})
        tmp=pd.merge(train,avg,on='shop_id',how='left')
        train['_'.join(['diff',cate,'today_d_6day'])]=tmp[cate]-tmp['shop_id_day6'+cate+'_avg']
    train.drop(col + item_cates + shop_cates, axis=1).to_csv('../data/'+writefilename,index=False)

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv') 
    trainday = 23
    user_continue_nobuy(org,trainday,'nobuy_feature_train.csv')
    trend_feature(org,trainday,'trend_feature_train.csv')
    oneshot_feature(org,trainday,'oneshot_feature_train.csv')
    first_ocr_feature(org,trainday,'ocr_feature_train.csv')
    item_shop_var_feature(org,trainday,'item_shop_var_feature_train.csv')
    trainday = 24
    user_continue_nobuy(org,trainday,'nobuy_feature_test.csv')
    trend_feature(org,trainday,'trend_feature_test.csv')
    oneshot_feature(org,trainday,'oneshot_feature_test.csv')
    first_ocr_feature(org,trainday,'ocr_feature_test.csv')
    item_shop_var_feature(org,trainday,'item_shop_var_feature_test.csv')
    
    
    