# coding=utf-8
# @author:bryan
"""
分别对于18-22\22日\18-23三个时间段，
计算各维度（用户、商品、品牌、店铺、类目、城市、二级类目、TOP10类目、预测类目属性、上下文ID、预测类目、预测一级类目）下的点击数，交叉点击数，交叉点击数占点击数比例

"""
import pandas as pd

def full_count_feature(org,name,trainday,writefilename):
    col=['user_id', 'item_id', 'item_brand_id', 'shop_id', 'item_category_list', 'item_city_id','cate','top10',
           'predict_category_property', 'context_page_id', 'query1', 'query']
    train=org[org.day==trainday][['instance_id']+col]
    if name=='day6':
        data = org[org.day==trainday-1][col]
    elif name=='days7':
        data=org[org.day<trainday][col] 
    elif name=='full':
        data=org[col]
    for item in col:
        train=pd.merge(train,data.groupby(item,as_index=False)['user_id'].agg({'_'.join([name,item,'cnt']):'count'}),on=item,how='left')
        print(item)
    items=col
    #计算两两交叉特征，算点击数
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            egg=[items[i],items[j]]
            #print(egg)
            tmp = data.groupby(egg, as_index=False)['user_id'].agg({'_'.join([name,items[i],items[j],'cnt']): 'count'})
            train = pd.merge(train, tmp, on=egg, how='left')
            print(egg)
    cross=[['user_id','query'],['user_id','query1'],['user_id','shop_id'],['user_id','item_id'],['item_id','shop_id'],['item_id', 'item_brand_id'],
           ['item_brand_id', 'shop_id'],['item_id','item_category_list'],['item_id','query'],
           [ 'item_id','item_city_id'],['item_id','cate'],['item_id','top10'],['item_id','context_page_id'],['item_id','query1'],
           ['item_brand_id', 'shop_id'],['shop_id','item_city_id'],[ 'shop_id','context_page_id']
           ]
    #指定一些组合，去算其中二元的占一元的比例，可以理解为结构占比
    for i in cross:
        train['_'.join(i+['cross'])]=train['_'.join([name,i[0],i[1],'cnt'])]/train['_'.join([name,i[1],'cnt'])]
        print(i)
    train=train.drop(col, axis=1)
    train.to_csv('../data/'+name+writefilename,index=False) 

if __name__ == '__main__':
    org=pd.read_csv('../data/origion_concat.csv')
    trainday = 23
    full_count_feature(org, 'day6',trainday,'_count_feature_train.csv')
    full_count_feature(org, 'days7',trainday,'_count_feature_train.csv')
    trainday = 24
    full_count_feature(org, 'day6',trainday,'_count_feature_test.csv')
    full_count_feature(org, 'days7',trainday,'_count_feature_test.csv') 
    