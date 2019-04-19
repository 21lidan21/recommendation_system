# coding=utf-8
# @author:bryan
"""
top100的特征强制相除交叉
""" 
import pandas as pd  

#观察可知这些表中重复的instance_id并不多，可以进行删除，否则merge后产生过多的行会占满内存
#    col=[query, leak, day6_cvr, days7_cvr,day6_rank, days7_rank, comp, nobuy, full,day6, days7, var, trend]
#    for i in col:print(i['instance_id'].value_counts().value_counts())

def select_unique_instanceid(org):
    tmp = org['instance_id'].value_counts()
    tmp = pd.DataFrame(tmp[tmp==1]);tmp.columns=['tmp']
    org=pd.merge(org,tmp,left_on='instance_id',right_index=True,how='inner')
    org.drop(['tmp'],axis=1,inplace=True)
    return org

if __name__ == '__main__': 
    for trainday in [23,24]:
        org=pd.read_csv('../data/origion_concat.csv')
        train=org[org.day==trainday]
        traintype = 'train' if trainday == 23 else  'test'
        query = pd.read_csv('../data/query_all_'+traintype+'.csv')
        leak = pd.read_csv('../data/leak_all.csv')
        comp = pd.read_csv('../data/compare_all.csv')
        day6_cvr = pd.read_csv('../data/6day_cvr_feature_'+traintype+'.csv')
        days7_cvr = pd.read_csv('../data/7days_cvr_feature_'+traintype+'.csv')
        day6_rank = pd.read_csv('../data/rank_feature_6day_'+traintype+'.csv')
        days7_rank = pd.read_csv('../data/rank_feature_7days_'+traintype+'.csv')
        nobuy = pd.read_csv('../data/nobuy_feature_'+traintype+'.csv')
        trend = pd.read_csv('../data/trend_feature_'+traintype+'.csv') 
        trend = trend[[i for i in trend.columns if 'cnt22' not in i]] #把cnt22的都去除掉
        var = pd.read_csv('../data/item_shop_var_feature_'+traintype+'.csv')
        user_buy_click = pd.read_csv('../data/user_buy_click_feature_'+traintype+'.csv')  
        day6 = pd.read_csv('../data/day6_count_feature_'+traintype+'.csv')
        days7 = pd.read_csv('../data/days7_count_feature_'+traintype+'.csv')
        # user_buy_click,property need proc caterory feature
        #去除重复的instance_id，否则在add函数使用的时候会产生大量的重复，乃至内存占满无法运行
        col=[query, leak, day6_cvr, days7_cvr,day6_rank, days7_rank, comp, nobuy, day6, days7, var, trend]
        data=select_unique_instanceid(train)
        for i in col:
            data=pd.merge(data,select_unique_instanceid(i),on='instance_id',how='left') 
        data.to_csv('../data/final_base_'+traintype+'.csv',index=False)  