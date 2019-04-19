# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 21:32:26 2019

@author: 59876
1、检索包含了用户意愿的重要信息，检索相关的特征如下：
本行数据的当天时间前后，此人有相同检索词的行数共有多少，有相同检索词 &相同的item_id\shop_id\brand_id\city_id\page_id\item_category_list的行数有多少，
有相同的检索词 & 不同的item_id的行数有多少，在这个检索词第一次出现前后的item_id\shop_id的item_id\shop_id个数有多少，
和本条item_id\shop_id\brand_id\city_id\cate一样的记录，之前\之后共有过多少个检索词，
前后的检索词个数占此人所有检索词个数的比例，
前后点击的item_id\shop_id\brand_id\city_id占此人所有点击的item_id\shop_id\brand_id\city_id的比例

2、间隔包含了用户的急缓、持续等信息，间隔相关的特征如下：
最大最小点击间隔，平均点击间隔，中位数点击间隔，上一个下一个间隔，最前最后一次间隔
（只有一条数据算-1,间隔均为秒数）

3、比较更低价优质的商品店铺包含了重要的理性购物信息，比较相关的特征如下：
之前之后点击了多少价格更低的商品，销量更高的商品，评价数更好的店铺（好评率高的店铺，星级高的店铺，服务态度高的店铺，物流好的店铺，描述平分高的店铺）

"""
from multiprocessing import Pool
from multiprocessing import cpu_count
import math
import pandas as pd 
import time
import numpy as np
import os 
import gc
import copy
os.getcwd()
   
##################################LK新版##
###修改原因：使用merge、lambda、groupby的组合编码方式，比原来使用循环的方式，计算缓慢，修改后的代码效率可以快约12倍
#CPU线程数-2
processor=cpu_count()-2
filenum = processor
#1、按照用户拆分成10分   开始
#按照10线程，将用户平均分为10分并且存到不同的文件里
def query_data_prepare(org):
    data=org 
    data = data.sort_values(by=['user_id', 'context_timestamp']).reset_index(drop=True)  
    users = pd.DataFrame(list(set(data['user_id'].values)), columns=['user_id'])  
    l_data = len(users)
    size = math.ceil(l_data / filenum)
    for i in range(filenum):
        start = size * i
        end = (i + 1) * size if (i + 1) * size < l_data else l_data
        user = users[start:end]
        t_data = pd.merge(data, user, on='user_id').reset_index(drop=True)
        t_data.to_csv('../data/user_data/query_'+str(i)+'.csv',index=False)
        print(len(t_data))
    t_data = pd.merge(data, users, on='user_id').reset_index(drop=True) 
    print(len(t_data))    


##
"""
    query特征,之前，之后有几次相同的query
    相同query，相同item，之前之后有多少个
    相同query,相同shop,之前之后个数
    相同query,相同brand,之前之后个数
    相同query,相同city,之前之后个数
    cate,page
    这个query之前之后是否搜过其他商品
    当前query之前之后点击了几个query
    """
def run_query_feature_like(i):
    data=pd.read_csv('../data/user_data/query_'+str(i)+'.csv') 
    col=['user_id','predict_category_property','context_timestamp','day','query1','query','item_id','shop_id','item_brand_id','item_city_id','context_page_id','item_category_list']
    data = data[['instance_id']+col] 
    
    col_choice1 = ['user_id','context_timestamp','day']
    col_choice2 = ['predict_category_property','query1','query']
    col_choice3 = ['item_id','shop_id','item_brand_id','item_city_id','context_page_id','item_category_list']
    col_choice2_map = {'predict_category_property':'query','query1':'query_1','query':'query_all'} 
    col_choice3_map = {'item_id':'item','shop_id':'shop','item_brand_id':'brand','item_city_id':'city','context_page_id':'page','item_category_list':'cate'}
    
    features=pd.DataFrame()
    
    for col2 in range(0,len(col_choice2)):    
        col_choice_final = col_choice2[col2:col2+1]+col_choice1  #;print(col_choice_final)
        tmp = pd.merge(data,data[col_choice_final],on=col_choice_final[0:2],how='left')
        print(' '.join(col_choice_final[0:2]) + ':merge done!')
        for i in ['before','after']: 
            if i == 'before':
                name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt'
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x','day_y','day_x','instance_id']].apply(lambda x:1 if (x[0]<x[1]) & (x[2]<=x[3]) else 0,axis=1)
                features[name] = tmp.groupby('instance_id')[name].sum() 
            else :
                name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt'
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x','day_y','day_x','instance_id']].apply(lambda x:1 if (x[0]>x[1]) & (x[2]>=x[3]) else 0,axis=1)
                features[name] = tmp.groupby('instance_id')[name].sum()
    
    for col3 in range(0,len(col_choice3)): 
        for col2 in range(0,len(col_choice2)):    
            col_choice_final = col_choice3[col3:col3+1]+col_choice2[col2:col2+1]+col_choice1  #;print(col_choice_final)
            tmp = pd.merge(data,data[col_choice_final],on=col_choice_final[0:3],how='left')
            print(' '.join(col_choice_final[0:3]) + ':merge done!')
            for i in ['before','after']: 
                if i == 'before':
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' + col_choice3_map.get(col_choice3[col3]) + '_' +'cnt'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x','day_y','day_x','instance_id']].apply(lambda x:1 if (x[0]<x[1]) & (x[2]<=x[3]) else 0,axis=1)
                    features[name] = tmp.groupby('instance_id')[name].sum() 
                else :
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' + col_choice3_map.get(col_choice3[col3]) + '_' +'cnt'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x','day_y','day_x','instance_id']].apply(lambda x:1 if (x[0]>x[1]) & (x[2]>=x[3]) else 0,axis=1)
                    features[name] = tmp.groupby('instance_id')[name].sum()
    
    for col2 in range(0,len(col_choice2)):    
        col_choice_final = col_choice2[col2:col2+1]+col_choice1  #;print(col_choice_final)
        tmp = pd.merge(data,data[col_choice_final],on='user_id',how='left')
        print(' '.join(col_choice_final[0:2]) + ':merge done!')
        for i in ['before','after']: 
            if i == 'before':
                name = i + '_diff_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt' #;print(name)
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x',col_choice2[col2]+'_y',col_choice2[col2]+'_x','instance_id']].apply(lambda x:1 if (x[0]<x[1]) & (x[2]!=x[3]) else 0,axis=1)
                features[name] = tmp.groupby('instance_id')[name].sum()
                
            else :
                name = i + '_diff_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt'
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x',col_choice2[col2]+'_y',col_choice2[col2]+'_x','instance_id']].apply(lambda x:1 if (x[0]>x[1]) & (x[2]!=x[3]) else 0,axis=1)
                features[name] = tmp.groupby('instance_id')[name].sum()
    
    from tkinter import _flatten 
    
    for col2 in range(0,len(col_choice2)):    
        col_choice_final = col_choice2[col2:col2+1]+col_choice1    
        tmp_time_before = data.groupby([col_choice2[col2],'user_id'])['context_timestamp'].agg({'context_timestamp_min':'min'}) 
        tmp_time_after  = data.groupby([col_choice2[col2],'user_id'])['context_timestamp'].agg({'context_timestamp_max':'max'}) 
        tmp_time = pd.merge(data[list(_flatten([col_choice_final,'instance_id','shop_id','item_id']))],tmp_time_before,left_on=col_choice_final[0:2],right_index=True,how='left')
        tmp_time = pd.merge(tmp_time,tmp_time_after,left_on=col_choice_final[0:2],right_index=True,how='left')
        tmp = pd.merge(tmp_time,tmp_time[list(_flatten([col_choice_final[0:2],'context_timestamp','shop_id','item_id']))],on='user_id',how='left') 
        print(' '.join(col_choice_final[0:2]) + ':merge done!')
        for k in ['shop_id','item_id']:  
            for i in ['before','after']: 
                if i == 'before': 
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +k.split('_')[0]+'s' #;print(name)
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_min','instance_id']].apply(lambda x:1 if x[0]<x[1] else 0,axis=1)
                    features[name] = tmp[['instance_id',k+'_y',name]].drop_duplicates().groupby('instance_id')[name].sum() 
                else :
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +k.split('_')[0]+'s' #;print(name)
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_max','instance_id']].apply(lambda x:1 if x[0]>x[1] else 0,axis=1)
                    features[name] = tmp[['instance_id',k+'_y',name]].drop_duplicates().groupby('instance_id')[name].sum()
    return features.reset_index()

def query_feature(writefilename):
    res = []
    p = Pool(processor)
    for i in range(filenum):
        res.append(p.apply_async(run_query_feature_like, args=( i,))) #p.apply_async(run_query_feature, args=( 0,));p.close();p.join()
        print(str(i) + ' processor started !')
    p.close()
    p.join()
    data=pd.concat([i.get() for i in res])
    data.to_csv('../data/'+writefilename,index=False)

"""
    最大最小点击间隔，平均点击间隔，只有一条数据算-1,上一个下一个间隔
    距离最前最后一次点击分钟数
    之前之后点击过多少query,item,shop,brand,city,query次数占比，item次数占比，shop,brand,city次数占比
    搜索这个商品,店铺，品牌，城市，用了几个query
    :param data:
    :return:
"""
def run_leak_feature_like(i):
    data=pd.read_csv('../data/user_data/query_'+str(i)+'.csv') 
    features=pd.DataFrame()
    
    col=['user_id','predict_category_property','context_timestamp','day','query1','query','item_id','shop_id','item_brand_id','item_city_id','context_page_id','item_category_list']
    data = data[['instance_id']+col] 
    data = data.drop_duplicates()
    
    data['context_timestamp_stamp'] = data['context_timestamp'].apply(lambda x:int(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))))
    data['ranktmp']=data[['context_timestamp_stamp','instance_id']].apply(lambda x:x[0]*100+x[1]*(10**-19),axis=1)
    data['timerankbyuserday']=data['ranktmp'].groupby(data[['user_id','day']].apply(lambda x:str(x[0])+str(x[1]),axis=1)).rank(method='min') 
    tmp=data[['user_id','context_timestamp_stamp','day','timerankbyuserday']] 
    tmp['timerankbyuserday'] = tmp['timerankbyuserday']-1
    tmp = tmp.set_index(['user_id','timerankbyuserday','day'])    
    data = pd.merge(copy.deepcopy(data).set_index(['user_id','timerankbyuserday','day']),copy.deepcopy(tmp),left_index=True,right_index=True,how='left').reset_index()
    data.rename(columns={'context_timestamp_stamp_x':'context_timestamp_stamp', 'context_timestamp_stamp_y':'context_timestamp_stamp_next'}, inplace = True)
    
    tmp=data[['user_id','context_timestamp_stamp','day','timerankbyuserday']] 
    tmp['timerankbyuserday'] = tmp['timerankbyuserday']+1  
    tmp = tmp.set_index(['user_id','timerankbyuserday','day'])    
    data = pd.merge(copy.deepcopy(data).set_index(['user_id','timerankbyuserday','day']),copy.deepcopy(tmp),left_index=True,right_index=True,how='left').reset_index()
    data.rename(columns={'context_timestamp_stamp_x':'context_timestamp_stamp','context_timestamp_stamp_y':'context_timestamp_stamp_previous'}, inplace = True)
    
    data['diffs'] = data['context_timestamp_stamp_next']-data['context_timestamp_stamp']    
    diffs = data.groupby(['user_id','day'])['diffs'].agg({'max_diff':'max','min_diff':'min','avg_diff':'mean','mid_diff':'median'})
    for i in diffs.columns: diffs[pd.isnull(diffs[i])]=-1
    stamps = data.groupby(['user_id','day'])['context_timestamp_stamp'].agg({'max_stamp':'max','min_stamp':'min','avg_stamp':'mean','mid_stamp':'median'})
    for i in stamps.columns: diffs[pd.isnull(stamps[i])]=-1
    
    features = pd.merge(data[['user_id','day','instance_id','context_timestamp_stamp','context_timestamp_stamp_previous','context_timestamp_stamp_next']],diffs,left_on=['user_id','day'],right_index=True,how='left') #;features.columns 
    features = pd.merge(features,stamps,left_on=['user_id','day'],right_index=True,how='left') #;features.columns 
    features=features.set_index(['instance_id'])    #;features.columns
    
    features['diff_first_click']=features['context_timestamp_stamp']-features['min_stamp']
    features['diff_last_click']=features['max_stamp']-features['context_timestamp_stamp']
    features['previous_diff']=features['context_timestamp_stamp']-features['context_timestamp_stamp_previous']
    features['next_diff']=features['context_timestamp_stamp_next']-features['context_timestamp_stamp']
    features.loc[pd.isnull(features['previous_diff']),'previous_diff']=-1
    features.loc[pd.isnull(features['next_diff']),'next_diff']=-1
 
    col_choice1 = ['user_id','day','context_timestamp']
    col_choice2 = ['predict_category_property','query1','query']
    col_choice3 = ['item_id','shop_id','item_brand_id','item_city_id','item_category_list']
    col_choice2_map = {'predict_category_property':'query','query1':'query_1','query':'query_all'} 
    col_choice3_map = {'item_id':'item','shop_id':'shop','item_brand_id':'brand','item_city_id':'city','context_page_id':'page','item_category_list':'cate'}
 
    for col3 in range(0,len(col_choice3)): 
        for col2 in range(0,len(col_choice2)):    
            col_choice_final = col_choice3[col3:col3+1]+col_choice1+col_choice2[col2:col2+1]  #;print(col_choice_final)
            tmp = pd.merge(data,data[col_choice_final],on=col_choice_final[0:3],how='left') #tmp.columns
            print('~'.join(col_choice_final[0:3]) + ':merge done!')
            for i in ['before','after']: 
                if i == 'before':
                    name = i + '_' + col_choice3_map.get(col_choice3[col3]) + '_' + col_choice2_map.get(col_choice2[col2])  + '_' +'cnt'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x', 'instance_id',col_choice2[col2]+'_y']].apply(lambda x:x[3] if (x[0]<=x[1]) else np.nan,axis=1)
                    features[name] = tmp[['instance_id',name]].drop_duplicates().groupby('instance_id')[name].count() 
                else :
                    name = i + '_' + col_choice3_map.get(col_choice3[col3]) + '_' + col_choice2_map.get(col_choice2[col2])  + '_' +'cnt'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x', 'instance_id',col_choice2[col2]+'_y']].apply(lambda x:x[3] if (x[0]>x[1]) else np.nan,axis=1)
                    features[name] = tmp[['instance_id',name]].drop_duplicates().groupby('instance_id')[name].count()
 
    for col3 in range(0,len(col_choice3)-1):  
            col_choice_final = col_choice1+col_choice3[col3:col3+1]   #;print(col_choice_final) 
            tmp = pd.merge(data,data[col_choice_final],on=col_choice_final[0:2],how='left') #tmp.columns
            print('~'.join(col_choice_final[0:2]) + ':merge done!')
            for i in ['before','after']: 
                if i == 'before':
                    name = i + '_' + col_choice3_map.get(col_choice3[col3]) + '_' +'rate'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x', 'instance_id',col_choice3[col3]+'_y']].apply(lambda x:x[3] if (x[0]<=x[1]) else np.nan,axis=1)
                    features[i + '_' + col_choice3_map.get(col_choice3[col3])] = tmp[['instance_id',name]].drop_duplicates().groupby(['instance_id'])[name].count()
                    countbypin=pd.DataFrame(tmp[['user_id','day',col_choice3[col3]+'_y']].drop_duplicates().groupby(['user_id','day'])[col_choice3[col3]+'_y'].count())
                    countbypin.columns=[col_choice3_map.get(col_choice3[col3])+'_'+i]
                    features = pd.merge(features,countbypin,left_on=['user_id','day'],right_index=True) 
                    features[name] = features[i + '_' + col_choice3_map.get(col_choice3[col3])]/features[col_choice3_map.get(col_choice3[col3])+'_'+i]
                else :
                    name = i + '_' + col_choice3_map.get(col_choice3[col3]) + '_' +'rate'   
                    features[name] = features['before' + '_' + col_choice3_map.get(col_choice3[col3]) + '_' +'rate'].apply(lambda x:1-x)
 
    for col2 in range(0,len(col_choice2)):  
            col_choice_final = col_choice1+col_choice2[col2:col2+1]   #;print(col_choice_final) 
            tmp = pd.merge(data,data[col_choice_final],on=col_choice_final[0:2],how='left') #tmp.columns
            print('~'.join(col_choice_final[0:2]) + ':merge done!')
            for i in ['before','after']: 
                if i == 'before':
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +'rate'  
                    tmp[name] = tmp[['context_timestamp_y','context_timestamp_x',col_choice2[col2]+'_y', 'instance_id']].apply(lambda x:x[2] if (x[0]<=x[1])  else np.nan,axis=1)
                    features[i + '_' + col_choice2_map.get(col_choice2[col2])] = tmp[['instance_id',name]].drop_duplicates().groupby(['instance_id'])[name].count()
                    countbypin=pd.DataFrame(tmp[['user_id','day',col_choice2[col2]+'_y']].drop_duplicates().groupby(['user_id','day'])[col_choice2[col2]+'_y'].count())
                    countbypin.columns=[col_choice2_map.get(col_choice2[col2])+'_'+i]
                    features = pd.merge(features,countbypin,left_on=['user_id','day'],right_index=True) 
                    features[name] = features[i + '_' + col_choice2_map.get(col_choice2[col2])]/features[col_choice2_map.get(col_choice2[col2])+'_'+i]
                else :
                    name = i + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +'rate'   
                    features[name] = features['before' + '_' + col_choice2_map.get(col_choice2[col2]) + '_' +'rate'].apply(lambda x:1-x)
    features=features[['after_brand_query_1_cnt', 'after_brand_query_all_cnt',
       'after_brand_query_cnt', 'after_brand_rate', 'after_cate_query_1_cnt',
       'after_cate_query_all_cnt', 'after_cate_query_cnt',
       'after_city_query_1_cnt', 'after_city_query_all_cnt',
       'after_city_query_cnt', 'after_city_rate', 'after_item_query_1_cnt',
       'after_item_query_all_cnt', 'after_item_query_cnt', 'after_item_rate',
       'after_query_1_rate', 'after_query_all_rate', 'after_query_rate',
       'after_shop_query_1_cnt', 'after_shop_query_all_cnt',
       'after_shop_query_cnt', 'after_shop_rate', 'avg_diff',
       'before_brand_query_1_cnt', 'before_brand_query_all_cnt',
       'before_brand_query_cnt', 'before_brand_rate',
       'before_cate_query_1_cnt', 'before_cate_query_all_cnt',
       'before_cate_query_cnt', 'before_city_query_1_cnt',
       'before_city_query_all_cnt', 'before_city_query_cnt',
       'before_city_rate', 'before_item_query_1_cnt',
       'before_item_query_all_cnt', 'before_item_query_cnt',
       'before_item_rate', 'before_query_1_rate', 'before_query_all_rate',
       'before_query_rate', 'before_shop_query_1_cnt',
       'before_shop_query_all_cnt', 'before_shop_query_cnt',
       'before_shop_rate', 'diff_first_click', 'diff_last_click', 'max_diff',
       'mid_diff', 'min_diff', 'next_diff', 'previous_diff']]
    return features.reset_index()

def leak_feature(writefilename):
    res = []
    p = Pool(processor)
    for i in range(filenum):
        res.append(p.apply_async(run_leak_feature_like, args=( i,)))
        print(str(i) + ' filenum started !')
    p.close()
    p.join()
    data = pd.concat([i.get() for i in res])
    data.to_csv('../data/'+writefilename,index=False)

"""
    当天的竞争特征
    之前之后点击了多少价格更低的商品，销量更高的商品，评价数更多的店铺，
    好评率高的店铺，星级高的店铺，服务态度高的店铺，物流好的店铺，描述平分高的店铺
    :param data:
    :return:
    """ 
def run_compare_feature_like(i):
    data=pd.read_csv('../data/user_data/query_'+str(i)+'.csv')  
    features=pd.DataFrame()
    
    col=['user_id','predict_category_property','context_timestamp','day','query1','query','item_id','shop_id','item_brand_id','item_city_id','context_page_id','item_category_list']
    col_choice2 = ['shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
    col_choice2_map = {'shop_review_num_level':'review_num','shop_review_positive_rate':'review_positive','shop_star_level':'star_level','shop_score_service':'score_service','shop_score_delivery':'score_delivery','shop_score_description':'score_description'}  

    data = data[['instance_id']+['item_price_level','item_sales_level']+col+col_choice2] 
    data = data.drop_duplicates()
    
    col_choice1 = ['user_id','day','context_timestamp']
    
    col_choice_final = col_choice1+['item_price_level','item_sales_level','item_id']
    tmp = pd.merge(data,data[col_choice_final],on=['user_id','day'],how='left')
    tmp['before_low_price_cnt'] = tmp[['context_timestamp_y','context_timestamp_x','item_price_level_y','item_price_level_x','item_id_y']].apply(lambda x:x[4] if (x[0]<x[1]) & (x[2]<x[3]) else np.nan,axis=1)
    tmp['after_low_price_cnt'] = tmp[['context_timestamp_y','context_timestamp_x','item_price_level_y','item_price_level_x','item_id_y']].apply(lambda x:x[4] if (x[0]>x[1]) & (x[2]<x[3]) else np.nan,axis=1)
    tmp['before_high_sale_cnt'] = tmp[['context_timestamp_y','context_timestamp_x','item_sales_level_y','item_sales_level_x','item_id_y']].apply(lambda x:x[4] if (x[0]<x[1]) & (x[2]>x[3]) else np.nan,axis=1)
    tmp['after_high_sale_cnt'] = tmp[['context_timestamp_y','context_timestamp_x','item_sales_level_y','item_sales_level_x','item_id_y']].apply(lambda x:x[4] if (x[0]>x[1]) & (x[2]>x[3]) else np.nan,axis=1)
    features['before_low_price_cnt'] = tmp[['instance_id','before_low_price_cnt']].drop_duplicates().groupby('instance_id')['before_low_price_cnt'].count()      
    features['after_low_price_cnt'] = tmp[['instance_id','after_low_price_cnt']].drop_duplicates().groupby('instance_id')['after_low_price_cnt'].count() 
    features['before_high_sale_cnt'] = tmp[['instance_id','before_high_sale_cnt']].drop_duplicates().groupby('instance_id')['before_high_sale_cnt'].count() 
    features['after_high_sale_cnt'] = tmp[['instance_id','after_high_sale_cnt']].drop_duplicates().groupby('instance_id')['after_high_sale_cnt'].count() 
 
    for col2 in range(0,len(col_choice2)):    
        col_choice_final = col_choice1+['shop_id']+col_choice2[col2:col2+1]  
        tmp = pd.merge(data,data[col_choice_final],on=['user_id','day'],how='left')
        print(' '.join(col_choice_final[0:2]) + ':merge done!')
        for i in ['before','after']: 
            if i == 'before':
                name = i + '_high_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt'  
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x',col_choice2[col2]+'_y',col_choice2[col2]+'_x','shop_id_y']].apply(lambda x:x[4] if (x[0]<x[1]) & (x[2]>x[3]) else np.nan,axis=1)
                features[name] = tmp[['instance_id',name]].drop_duplicates().groupby('instance_id')[name].count()      
            else :
                name = i + '_high_' + col_choice2_map.get(col_choice2[col2]) + '_' +'cnt'
                tmp[name] = tmp[['context_timestamp_y','context_timestamp_x',col_choice2[col2]+'_y',col_choice2[col2]+'_x','shop_id_y']].apply(lambda x:x[4] if (x[0]>x[1]) & (x[2]>x[3]) else np.nan,axis=1)
                features[name] = tmp[['instance_id',name]].drop_duplicates().groupby('instance_id')[name].count()    
    return features.reset_index()

def compare_feature(writefilename): 
    res = []
    p = Pool(processor)
    for i in range(filenum):
        res.append(p.apply_async(run_compare_feature_like, args=(i,)))
        print(str(i) + ' filenum started !')
    p.close()
    p.join()
    data = pd.concat([i.get() for i in res])
    data.to_csv('../data/'+writefilename,index=False) 

if __name__ == '__main__':  
    org = pd.read_csv('../data/origion_concat.csv')
    trainday = 24
    org = org[(org['day']<=trainday) & (org['day']>=trainday-5)]
    query_data_prepare(org)
    gc.collect()
    start_time = time.time()
    print("start_time ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    query_feature('query_all_test.csv')
    print('query_feature_test finish')
    print("cost {d} seconds".format(d = int(time.time()-start_time)))
    
    org = pd.read_csv('../data/origion_concat.csv')
    trainday = 23
    org = org[(org['day']<=trainday) & (org['day']>=trainday-5)]
    query_data_prepare(org)
    gc.collect()
    start_time = time.time()
    print("start_time ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    query_feature('query_all_train.csv')
    print('query_feature_train finish')
    print("cost {d} seconds".format(d = int(time.time()-start_time)))
    
    org = pd.read_csv('../data/origion_concat.csv') 
    query_data_prepare(org)
    gc.collect()
    start_time = time.time()
    leak_feature('leak_all.csv')
    print('leak_feature finish')
    print("cost {d} seconds".format(d = int(time.time()-start_time)))
    start_time = time.time()
    compare_feature('compare_all.csv')
    print('compare_feature finish')  
    print("cost {d} seconds".format(d = int(time.time()-start_time)))
    start_time = time.time()
    print("end_time ",time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))  
    
 