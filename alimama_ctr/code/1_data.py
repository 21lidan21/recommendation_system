# -*- coding: utf-8 -*- 
'''
相关背景——
赛题背景：搜索广告，商家（广告主）根据商品特点自主购买特定的关键词，当用户输入这些关键词时相应的广告商品就会展示在用户看到的页面中
阿里目前已经采用的技术方向：深度学习、在线学习、强化学习
五大类数据信息，用户、广告商品、检索词、上下文内容、商店
1、用户：ID、性别、年龄、职业、星级
2、广告商品：ID、类目、属性、品牌、城市、价格等级、销量等级、收藏次数、展示次数
3、检索词：根据检索词预测的类目属性列表
4、上下文内容：ID、与当下场景相关的信息，包括广告商品的展示时间、广告商品的曝光屏幕编号
5、店铺信息：ID、评价数量等级、好评率、星级、服务态度评分、物流服务评分、描述相符评分

八大类特征：
1、相同\前后检索时间维度：预测日前1天\前5天，前后相同检索词的行数、商品数等等，前后相同商品等的的检索词数，前后的商品数等
2、单日间隔秒数维度：点击前后的间隔、最大最小间隔、平均中位数间隔
3、单日比价比质维度：在购买前，更低价格更好店铺的浏览
4、转化率时间维度：预测日前1天\前5天的转化率特征，做为预测日的特征，包括用户商品等等，及一些二维交叉转化度，还有某些维度在另一些维度下转化率的排名
5、购买\未购买时间维度：预测日前1天\前5天的最大连续未购买点击数，购买\未购买商品\店铺数，各维度下的点击\购买数及后一天比前一天的比例，几个重要维度下的
    即时购买率（每次点击均购买），第一次点击到第一次购买的时间间隔，用户点击的商品热度\价格的按日变化，用户点击的店铺评分的按日变化
6、重要特征时间热度：计算商品\品牌等等重要维度预测日在前1天\预测日前5天的点击数，及这些维度的交叉维度的点击数
7、单次点击&搜索匹配属性：点击商品属性和检索词预测属性\类目一致数目；点击商品属性\类别个数；搜索词预测属性\类别个数；点击商品属性里TOP热度的属性
8、top特征交叉相除：对重要性TOP100特征进行两两交叉相除，之后从相除的特征里再次选出重要性TOP100做为新特征（重要性以LGB计算）

特征明细：
1、相同\前后检索维度：
本行数据的当天时间前后，此人有相同检索词的行数共有多少，有相同检索词 &相同的item_id\shop_id\brand_id\city_id\page_id\item_category_list的行数有多少，
有相同的检索词 & 不同的item_id的行数有多少，在这个检索词第一次出现前后的item_id\shop_id的item_id\shop_id个数有多少，
和本条item_id\shop_id\brand_id\city_id\cate一样的记录，之前\之后共有过多少个检索词，
前后的检索词个数占此人所有检索词个数的比例，
前后点击的item_id\shop_id\brand_id\city_id占此人所有点击的item_id\shop_id\brand_id\city_id的比例
2、间隔时间维度：
最大最小点击间隔，平均点击间隔，中位数点击间隔，上一个下一个间隔，最前最后一次间隔
（只有一条数据算-1,间隔均为秒数）
3、比价比质维度：
之前之后点击了多少价格更低的商品，销量更高的商品，评价数更好的店铺（好评率高的店铺，星级高的店铺，服务态度高的店铺，物流好的店铺，描述平分高的店铺）
4、转化率时间维度：
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
5、购买\未购买时间维度：
之后的没有整理....在各自的py文件里有 
''' 

import pandas as pd  
##############1、导入数据 
data = pd.read_csv('../data/round1/round1_ijcai_18_train_20180301.txt', sep=' ')   

##############2、增加时间字段  开始
import time
data['day'] = data['context_timestamp'].apply(lambda x:time.localtime(x)[2]) #data['day'].value_counts()
data['hour'] = data['context_timestamp'].apply(lambda x:0 if time.localtime(x)[3]==31 else time.localtime(x)[3]) 

def today(x):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x))

data['context_timestamp'] = data['context_timestamp'].apply(lambda x:today(x))

def gethour(x):
    hour=int(x.split(' ')[1].split(':')[0])
    minute=int(x.split(' ')[1].split(':')[1])
    minute=1 if minute>=30 else 0
    return hour*2+minute

data['hour48']=data['context_timestamp'].apply(gethour) 
##############增加时间字段  结束

################3、取出item_category_list和predict_category_property中一样的类目的个数 开始
def same_cate(x):
    cate = set(x['item_category_list'].split(';'))
    cate2 = set([i.split(':')[0] for i in x['predict_category_property'].split(';')])
    return len(cate & cate2) 

data['same_cate']=data.apply(same_cate,axis=1) #相同类别数  

##############4、item_property_list和predict_category_property相同属性数计算  开始 
def same_property(x):
    property_a = set(x['item_property_list'].split(';'))
    a = []
    for i in [i.split(':')[1].split(',') for i in x['predict_category_property'].split(';') if
              len(i.split(':')) > 1]:
        a += i
    property_b = set(a)
    return len(property_a & property_b)

data['same_property']=data.apply(same_property,axis=1) #相同属性数
#######################item_property_list和predict_category_property相同属性数计算结束 

data['property_num']=data['item_property_list'].apply(lambda x:len(x.split(';'))) #属性的数目
data['pred_cate_num']=data['predict_category_property'].apply(lambda x:len(x.split(';'))) #query的类别数目

###############5、计算predict_category_property的属性取值数目开始 
from functools import reduce
def f(x):
        try:
            return len([i for i in reduce((lambda x, y: x + y), [i.split(':')[1].split(',') for i in x.split(';') if len(i.split(':'))>1]) if i != '-1'])
        except:
            return 0
data['pred_prop_num']=data['predict_category_property'].apply(f) #query的属性取值的数目，不包含-1
##############计算predict_category_property的属性取值数目结束
 
data['query1']=data['predict_category_property'].apply(lambda x:x.split(';')[0].split(':')[0]) #query第一个类目 
#predict_category_property 所有类目
data['query']=data['predict_category_property'].apply(lambda x:'-'.join(sorted([i.split(':')[0] for i in x.split(';')]))) 
data['cate'] = data['item_category_list'].apply(lambda x: x.split(';')[1]) #选择二级类目

##################6、填充缺失值开始 
def fillna(data):
    numeric_feature = ['day', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level',
                       'user_age_level', 'user_star_level', 'shop_review_num_level',
                       'shop_review_positive_rate', 'shop_star_level', 'shop_score_service', 'shop_score_delivery',
                       'shop_score_description', 'context_page_id'
                       ] 
    string_feature = ['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'user_gender_id',
                      'user_occupation_id', 'context_page_id', 'hour'] 
    other_feature = ['item_property_list', 'predict_category_property'] 
    #填充缺失值
    for i in string_feature+other_feature:
        mode_num = data[i].mode()[0]   
        if (mode_num != -1):
            print(i)
            data.loc[data[i] == -1, i] = mode_num
        else:
            print(-1)
    for i in numeric_feature:
        mean_num = data[i].mean()
        if (mean_num != -1):
            print(i)
            data.loc[data[i] == -1, i] = mean_num
        else:
            print(-1)
    return data 
data=fillna(data.copy())
####################填充缺失值结束

##################7、拆分属性 开始
"""
统计属性出现的次数，取top1的属性作为特征，top1-5合并作为特征
预测的属性，top1,合并top1-5
"""

def property_feature(org):
    tmp=org['item_property_list'].apply(lambda x:x.split(';')).values
    property_dict={}
    property_list=[] 
    for i in tmp:  
        property_list+=i   
    for i in property_list:   
        if i in property_dict:
            property_dict[i]+=1   
        else:
            property_dict[i] = 1
    print('dict finish')
    def top(x):
        propertys=x.split(';')
        cnt=[property_dict[i] for i in propertys]  
        res=sorted(zip(propertys,cnt),key=lambda x:x[1],reverse=True)  
        top1=res[0][0]
        top2 = '_'.join([i[0] for i in res[:2]])
        top3 = '_'.join([i[0] for i in res[:3]])
        top4 = '_'.join([i[0] for i in res[:4]])
        top5='_'.join([i[0] for i in res[:5]])
        top10 = '_'.join([i[0] for i in res[:10]])
        return (top1,top2,top3,top4,top5,top10)
    org['top']=org['item_property_list'].apply(top)
    print('top finish')
    org['top1']=org['top'].apply(lambda x: x[0])
    org['top2'] = org['top'].apply(lambda x: x[1])
    org['top3'] = org['top'].apply(lambda x: x[2])
    org['top4'] = org['top'].apply(lambda x: x[3])
    org['top5'] = org['top'].apply(lambda x: x[4])
    org['top10'] = org['top'].apply(lambda x: x[5])
    return org[['instance_id','top1','top2','top3','top4','top5','top10']] 

data=pd.merge(data,property_feature(data.copy()),on='instance_id',how='left') #拆分属性 
##################拆分属性结束

#################8、类别特征全部编码开始 
from sklearn.preprocessing import LabelEncoder
def encode(data):
    id_features=['shop_id', 'item_id', 'user_id', 'item_brand_id', 'item_city_id', 'user_gender_id','item_property_list', 'predict_category_property',
                      'user_occupation_id', 'context_page_id','top1','top2','top3','top4','top5','top10','query1','query','cate']
    for feature in id_features:
        data[feature] = LabelEncoder().fit_transform(data[feature])
    return data 

data=encode(data)
###############类别特征全部编码结束

data.to_csv('../data/origion_concat.csv',index=False) 

