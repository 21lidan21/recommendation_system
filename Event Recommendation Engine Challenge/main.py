import pandas as pd
from collections import defaultdict
import scipy.sparse as ss
import scipy.io as sio
import itertools
import _pickle as cPickle


# if __name__=="__main__":   
# 一 查看缺失值状态
# train和test都不存在缺失数据
df_train=pd.read_csv('./data/train.csv')
df_test=pd.read_csv('./data/test.csv')
users=pd.read_csv('./data/users.csv')
events=pd.read_csv('./data/events.csv')
# # print(df_train.head())
print(events.columns)
print('train 缺失状态：')
# print()
# print(pd.DataFrame(df_train.isnull().sum()).T)
print(pd.DataFrame(users.isnull().sum()).T)
# 二 数据清洗
# 四大方面的工作： 1）文本数据====>数值型数据(如：gender) 2）时间数据====>数值型数据(如：jointedAt) 3）地理位置数据====>数值型数据(如：location) 4) 多字段数据 ===>数值型数据
#1定义gender
def gender_to_data(genderstr):
    if genderstr=='male':
        return 1
    elif genderstr == 'famale':
        return 2
    else:
        return 0
#2定义birth
def birth_to_int(birthyear):
    if birthyear == 'None':
        return 0
    else:
        return int(birthyear)
#3定义timezone 
def timezone_to_int(timezonestr):
    try:
        return int(timezonestr)
    except:
       return 0
#4定义年月  
def get_year_month(dateString):
    dttm=''.join(dateString[:8].strip('"').split('-'))
    return dttm
#locale===>data
locale_id=defaultdict(int)
# enumerate 可循环遍历对象
for i,l in enumerate(locale_alias.keys()):
    locale_id[l]=i+1

def locale_to_data(localestr):
    return locale_id[localestr.lower()]

#loacation===>data
country_id = defaultdict(int)
city_id=defaultdict(int)
for i,c in enumerate(pycountry.countries):
    country_id[c.name.lower()]=i+1
    if c.name.lower()=='usa':
        ctry_id['US']=i
    if c.name.lower()=='canada':
        ctry_id['CA']=i
for cc in ctry_id.keys():
    for s in pycountry.subdivisions.get(country_code==cc):
        country_id[s.name.lower()]=ctry[cc]+1

def get_country_id(location):
    if((isinstance(location,str)) and len(location.strip())>0 and location.rfind('')>-1):
        return country_id[location[location.rindex('')+2:].lower()]
    else:
        return 0

def get_feature_hash(value):
    if len(value.strip())==0:
        return -1
    else:
        return int(hashlib.sha224(value.encode('utf-8')).hexdigest()[0:4],16)

def get_float_value(value):
    if len(value.strip())==0 or value =='NA':
        return 0.0
    else:
        return float(value)

# 2.处理user和event关联数据¶
class ProgramEntities:
    def _init_(self):
        uniqueUsers = set()
        uniqueEvents=set()
        eventsForUser=defaultdict(set)
        userForEvent = defaultdict(set)
        for filename in ['./data/train.csv','./data/test.csv']:
            f= open(filename)
            f.readline()#表示要开始读第一行
            for line in f:
                cols= line.strip().split(',')
                uniqueUsers.add(cols[0])
                uniqueEvents.add(cols[1])
                print('uniqueUsers:')
                print(uniqueUsers)
                eventsForUser[cols[0]].add(cols[1])#某个用户的events集合量
                userForEvent[cols[1]].add(cols[0])#某个event的用户集合量
            f.close()
        print('训练集和测试集合计独立用户的数量是：',len(uniqueUsers))
        print('训练集和测试集合计独立用户的数量是：',len(uniqueEvents))
        #构建user—event得分矩阵
        #字典形式存取稀疏矩阵，定义行和列的数量
        self.userEventScores = ss.dok_matrix(len(uniqueUsers),len(uniqueEvents))
        self.userIndex=dict()
        self.eventIndex=dict()
        for i,u in enumerate(uniqueUsers):
            self.userIndex[u]=i
        for i,e in enumerate(uniqueEvents):
            self.eventIndex[e]=i
        ftrain=open("./data/train.csv",'rb')
        # user,event,invited,timestamp,interested,not_interested
        ftrain.readline()
        for line in ftrain:
            cols= line.strip().split(',')
            i= self.userIndex[cols[0]]
            j= self.eventIndex[cols[1]]
            self.userEventScores[i,j]=int(cols[4])-int(cols[5])
        ftrain.close()
        sio.mmwrite("PE_userEventScores",self.userEventScores)

        # 为了防止不必要的计算，我们找出来所有关联的用户 或者 关联的event
        # 所谓的关联用户，指的是至少在同一个event上有行为的用户pair
        # 关联的event指的是至少同一个user有行为的event pair
        # 相似度至少应该发生在两个元素之间，在下面设定长度要>2
        self.uniqueUserPairs=set()
        self.uniqueEventPairs=set()
        for event in uniqueEvents:
            users= userForEvent[event]
            if len(users)>2:
                self.uniqueUserPairs.update(itertools.combinations(users,2))
        for user in uniqueUser:
            events= eventsForUser[user]
            if len(events)>2:
                self.uniqueEventPairs.update(itertools.combinations(events,2))
        cPickle.dump(self.userIndex,open('PE_userIndex.pkl', 'wb'))
        cPickle.dump(self.eventIndex,open("PE_eventIndex.pkl", 'wb'))

# 3.用户与用户相似度矩阵
class Users:
    def _int_(self,ProgramEntities,sim=ssd.correlation):
        cleaner=DataCleaner()
        nusers=len(ProgramEntities.userIndex.keys())#计算user总数量 3391 train 和 test合计出现的总的users数量
        fin = open('./data/users.csv','rb')
        # user_id,locale,birthyear,gender,joinedAt,location,timezone 7
        colnames=fin.readline().strip().split(',')
        self.userMatrix=ss.dok_matrix((nusers,len(colnames)-1))
        for line in fin:
            cols=line.strip().split(',')
            if ProgramEntities.userIndex.has_key(cols[0]):
                i = ProgramEntities.userIndex[cols[0]]
                self.userMatrix[i,0]=cleaner.getLocaleId(cols[1])
                self.userMatrix[i,1]=cleaner.getBirthYearInt(cols[2])
                self.userMatrix[i,2]=cleaner.getGenderId(cols[3])
                self.userMatrix[i,3]=cleaner.getJoinedYearMonth(cols[4])
                self.userMatrix[i,4]=cleaner.getCountry(cols[5])
                self.userMatrix[i,5]=cleaner.getTimezoneInt(cols[6])
            fin.close()
            # 归一化用户矩阵
            self.userMatrix=normalize(self.userMatrix,norm='l1',axis=0,copy=False)
            sio.mmwrite("US_userMatrix",self.userMatrix)
             # 计算用户相似度矩阵，之后会用到
            self.userSimMatrix=ss.dok_matrix((nusers,nusers))
            for i in range(0,nusers):
                self.userSimMatrix[i,i]=1.0
            for u1,u2 in ProgramEntities.uniqueUserPairs:
                i= ProgramEntities.userIndex[u1]
                j= ProgramEntities.userIndex[u2]
                if not self.userSimMatrix.haskey((i,j)):
                    # 用户的相似度计算这里采用pearson相关距离公式计算
                    usim=sim(self.userMatrix.getrow(i).todense(),self.userMatrix.getrow(j).todense())
                    self.userSimMatrix[i,j]=usim
                    self.userSimMatrix[j,i]=usim
            sio.mmwrite("US_userSimMatrix", self.userSimMatrix)
# 4.用户社交关系挖掘
class UserFriends:
    def _int_(self,ProgramEntities):
        nusers = len(ProgramEntities.userIndex.keys())
        self.numFriends=np.zeros((nusers))
        self.UserFriends=ss.dok_matrix((nusers,nusers))
        fin = open("./data/user_friends.csv",'rb')
        fin.readline()
        ln=0
        for line in fin:
            if ln % 200 ==0:
                print("Loading line:",ln)
            cols = line.strip().split(',')
            user =cols[0]
            if ProgramEntities.userIndex.has_key(user):
                friends=cols[1].split(" ")
                i=ProgramEntities.userIndex[user]
                self.numFriends[i]=len(friends)
                for friend in friends:
                    if ProgramEntities.userIndex.has_key(friend):
                        j=ProgramEntities.userIndex[friend]
                        eventsForUser = ProgramEntities.userEventScores.getrow(j).todense()
                        score =eventsForUser.sum()/np.shape(eventsForUser)[1]
                        self.UserFriends[i,j]+=score
                        self.UserFriends[j,i]+=score
        ln +=1
    fin.close()
    # 归一化数组
    sumNumFriends = self.numFriends.sum(axis=0)
    self.numFriends=self.numFriends/sumNumFriends
    sio.mmwrite("UF_numFriends",np.matrix(self.numFriends))
    self.UserFriends=normalize(self.UserFriends,norm='l1',axis=0,copy=False)
    sio.mmwrite("UF_userFriends",self.UserFriends)
# 5.构造event和event相似度数据
class Events:
    def _int_(self,ProgramEntities,psim=ssd.correlation,csim=ssd.cosine):
        cleaner = DataCleaner()
        fin = open('./data/users.csv','rb')
        fin.readline()
        nevents = len(ProgramEntities.eventIndex.keys())
        self.eventPropMatrix =ss.dok_matrix((nevents,7))
        self.eventConMatrix=ss.dok_matrix((nevents,100))
        ln=0
        for line in fin.readlines():
            if ln>10:
                break
            cols=line.strip().split(",")
            eventId=cols[0]
            if ProgramEntities.eventIndex.has_key(eventId):
                i=ProgramEntities.eventIndex[eventId] 
                self.eventPropMatrix[i,0]=cleaner.getJoinedYearMonth(cols[2])
                self.eventPropMatrix[i,1]=cleaner.getFeatureHash(cols[3])  
                self.eventPropMatrix[i,2]=cleaner.getFeatureHash(cols[4])
                self.eventPropMatrix[i,3]=cleaner.getFeatureHash(cols[5])
                self.eventPropMatrix[i,4]=cleaner.getFeatureHash(cols[6])
                self.eventPropMatrix[i,5]=cleaner.getFloatValue(cols[7])
                self.eventPropMatrix[i,6]=cleaner.getFloatValue(cols[8])
                for j in range(9,109):
                    self.eventConMatrix[i,j-9]=cols[j]
                ln +=1
            fin.close()
            self.eventPropMatrix=normalize(self.eventPropMatrix,norm='l1',axis=0,copy=False)
            sio.mmwrite("EV_eventPropMatrix",self.eventPropMatrix)
            self.eventConMatrix=normalize(self.eventConMatrix,norm='l1',axis=0,copy=False)
            sio.mmwrite("EV_eventContMatrix",self.eventConMatrix)
            # 计算活动对的相似度
            self.eventPropSim=ss.dok_matrix((nevents,nevents))
            self.eventContSim=ss.dok_matrix((nevents,nevents))
            for e1,e2 in ProgramEntities.uniqueEventPairs:
                i=ProgramEntities.eventIndex[e1]
                j=ProgramEntities.eventIndex[e2]
                if not self.eventPropSim.has_key(i,j):
                    epsim=psim(self.eventPropMatrix.getrow(i).todense(),self.eventPropMatrix.getrow(j).todense())
                    self.eventPropSim[i,j]=epsim
                    self.eventPropSim[j,i]=epsim
                if not self.eventConMatrix.has_key((i,j)):
                    ecsim=csim(self.eventConMatrix.getrow(i).todense(),self.eventConMatrix.getrow(j).todense())
                    self.eventContSim[i,j]=ecsim
                    self.eventContSim[j,i]=ecsim
            sio.mmwrite("EV_eventPropSim",self.eventPropSim)
            sio.mmwrite("EV_eventContSim",self.eventContSim)
# 6.活跃度/event热度 数据
class EventAttendees():
    def _init_(self,ProgramEntities):
        nevents = len(programEvents.eventIndex.keys())
        self.eventPopularity = ss.dok_matrix((nevents,1))
        f=open("./data/event_attendees.csv", 'rb')
        f.readline()
        for line in f:
            cols=line.strip().split(',')
            eventId =cols[0]
            if programEvents.eventIndex.has_key(eventId):
                i=programEvents.eventIndex[eventId]
                self.eventPopularity[i,0]=len(cols[1].split(' '))-len(cols[4].split(' '))
        f.close()
        self.eventPopularity=normalize(self.eventPopularity,norm='l1',axis=0,copy=False) 
        sio.mmwrite("EA_eventPopularity",self.eventPopularity)   

#7.串起所有的数据处理和准备流程
def data_prepare():
     print("第1步：统计user和event相关信息...")
     pe = ProgramEntities() 
     print("第1步完成...\n") 
     print("第2步：计算用户相似度信息，并用矩阵形式存储...")
     Users(pe)
     print("第2步完成...\n")
     print("第3步：计算用户社交关系信息，并存储...")
     UserFriends(pe)
     print("第3步完成...\n")
     print("第4步：计算event相似度信息，并用矩阵形式存储...")
     Events(pe)
     print("第4步完成...\n")
     print("第5步：计算event热度信息...")
     EventAttendees(pe)
     print("第5步完成...\n")
# 运行进行数据准备
data_prepare()
# 8.构建特征
# 这是构建特征部分
from __future__ import division

import cPickle
import numpy as np
import scipy.io as sio

class DataRewriter:
    def _int_(self)
        self.userIndex= cPickle.load(open("PE_userIndex.pkl",'rb'))
        self.eventIndex= cPickle.load(open("PE_eventIndex.pkl", 'rb'))
        self.userEventScores=sio.mmread("PE_userEventScores").todense()
        self.userSimMatrix=sio.mmread("US_userSimMatrix").todense()
        self.eventPropSim=sio.mmread("EV_eventPropSim").todense()
        self.eventConSim=sio.mmread("EV_eventConSim").todense()
        self.numFriends = sio.mmread("UF_numFriends")
        self.userFriends = sio.mmread("UF_userFriends").todense()
        self.eventPopularity = sio.mmread("EA_eventPopularity").todense()
    def userReco(self,userId,eventId):
        i=self.userIndex[userId]
        j=self.eventIndex[eventId]
        vs = self.userEventScores[:,j]
        sims=self.userSimMatrix[i,:]
        prod=sims*vs
        try:
            return prod[0,0]-self.userEventScores[i,j]
        except IndexError:
            return 0
    def eventReco(self,userId,eventId):
        # 根据基于物品的协同过滤，得到Event的推荐度
        i=self.userIndex[userId]
        j=self.eventIndex[eventId]
        js=self.userEventScores[i,:]
        psim=self.eventPropSim[:,j]
        csim=self.eventContSim[:,j]
        pprod = js*psim
        cprod = js*csim
        pscore = 0
        cscore = 0
        try:
            pscore =pprod[0,0]-self.userEventScores[i,j]
        except IndexError:
            pass
        try:
            cscore=cprod[0,0]-self.userEventScores[i,j]
        except IndexError:
            pass
        return pscore,cscore
    def userPop(self,userId):
    # 基于用户的朋友个数来推断用户的社交程度
        if self.userIndex.has_key(userId):
            i = self.userIndex[userId]
            try:
                return self.numFriends[0,i]
            except IndexError:
                return 0
        else:
            return 0  
    def friendInfluence(self, userId):  
        # 朋友对用户的影响
        nusers=np.shape(self.userFriends)[1]
        i=self.userIndex[userId]  
        return (self.userFriends[i,:].sum(axis=0)/nusers)[0,0] 
    def eventPop(self,eventId):
          """
    本活动本身的热度
    主要是通过参与的人数来界定的
    """
        i = self.eventIndex[eventId]
        return self.eventPopularity[i,0]
    def rewriteData(self,start=1,train=True,header=True):
        """
    把前面user-based协同过滤 和 item-based协同过滤，以及各种热度和影响度作为特征组合在一起
    生成新的训练数据，用于分类器分类使用
    """
        fn="./data/train.csv" if train else "./data/test.csv"
        fin = open(fn,"rb")
        fout =open("data_"+fn,'wb')
        if header:
            ocolnames=["invited","user_reco","evt_p_reco","evt_c_reco","user_pop","frnd_inf1","evt_pop"]
            if train:
                ocolnames.append("interested")
                ocolnames.append("not_interested")
            fout.write(",".join(ocolnames)+'\n')
        ln=0
        for line in fin:
            ln+=1
            if ln<start:
                continue
            cols=line.strip().split(',')
            userId=cols[0]
            eventId=cols[1]
            invited=cols[2]
            if ln%500 ==0:
                print("%s:%d (userId,eventId)=(%s,%s)"%(fn,ln,userId,eventId))
            user_reco=self.userReco(userId,eventId)
            evt_p_reco,evt_c_reco=self.eventReco(userId,eventId)
            user_pop=self.userPop(userId)
            frnd_infl=self.friendInfluence(userId)
            evt_pop=self.eventPop(eventId)
            ocols=[invited,user_reco,evt_p_reco,evt_c_reco,user_pop,frnd_infl,evt_pop]
            if train
                ocols.append(cols[4])
                ocols.append(cols[5])
            fout.write(",".join(map(lambda x:str(x),ocols))+'\n')
        fin.close()
        fout.close()
    
    def rewriteTrainingSet(self):
        self.rewriteData(True)
    def rewriteTestSet(self):
        self.rewriteData(False)
dr =DataRewriter()
print("生成训练数据...\n")
dr.rewriteData(train=True,start=2,header=True)
print("生成预测数据...\n")
dr.rewriteData(train=False,start=2,header=True)
# 9.建模与预测
from __future__ import division
import math
import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.linear_model import SGDClassifier

def train():
    """
  在我们得到的特征上训练分类器，target为1(感兴趣)，或者是0(不感兴趣)
  """
    trainDf=pd.read_csv("./data/data_train.csv")
    X = np.matrix(pd.DataFrame(trainDf,index=None,columns=["invited","user_reco","evt_p_reco","evt_c_reco","user_pop","frnd_infl","evt_pop"]))
    y=np.array(trainDf.interested)
    clf=SGDClassifier(loss="log",penalty='l2')
    clf.fit(X,y)
    return clf

def validate():
    """
  10折的交叉验证，并输出交叉验证的平均准确率
  """
  trainDf=pd.read_csv("./data/data_train.csv")
  X=np.matrix(pd.DataFrame(trainDf,index=None,columns=["invited","user_reco","evt_p_reco","evt_c_reco","user_pop","frnd_infl","evt_pop"]))
  y =np.array(trainDf.interested)
  nrows=len(trainDf)
  Kfold=KFold(nrows,10)
  avgAccuracy = 0
  run=0
  for train,test in Kfold:
      Xtrain,Xtest,ytrain,ytest=X[train],X[test],y[train],y[test]
      clf=SGDClassifier(loss='log',penalty='l2')
      clf.fit(Xtrain,ytrain)
      accuracy=0
      ntest=len(ytest)
      for i in range(0,ntest):
          yt=clf.predict(Xtest[i,:])
          if yt==ytest[i]:
              accuracy+=1
      accuracy=accuracy/ntest
      print("accuracy(run %d):%f" % (run,accuracy))
      avgAccuracy+=accuracy
      run +=1
      print("Average accuracy", (avgAccuracy / run))
def test(clf):
    oriTestDf=pd.read_csv('./data/test.csv')
    users=oriTestDf.user
    events=oriTestDf.event
    testDf=pd.read_csv('./data/data_test.csv')
    fout=open("result.csv",'wb')
    fout.write(','.join(['user','event','outcome','dist'])+'\n')
    nrows=len(testDf)
    Xp=np.matrix(testDf)
    yp=np.zeros((nrows,2))
    for i in range(0,nrows):
        xp=Xp[i,:]
        yp[i,0]=clf.predict(xp)
        yp[i,1]=clf.decision_function(xp)
        fout.write(",".join(map(lambda x:str(x),[users[i],events[i],yp[i,0],yp[i,1]]))+'\n')
    fout.close()
clf=train()
test(clf)
# 10.生成要提交的文件

def byDist(x,y):
    return int(y[1]-x[1])
def generate_submition_file():
    # 输出文件
    fout =open("final_result.csv", 'wb')
    fout.write(','.join(['User','Events'])+'\n')
    resultDf=pd.read_csv('result.csv')
    grouped=resultDf.groupby('user')
    for name,group in grouped:
        user=str(name)
        tuples=zip(list(group.event),list(group.dist),list(group.outcome))
    fout.close()
generate_submition_file()      

from sklearn.learning_curve import learning_curve
clf =LogisticRegression(penalty="l2")
train_sizes,train_scores,test_scores = learning_curve(estimator=clf,X=X,
                                                     y=y,
                                                     train_sizes=np.linspace(0.05,1.0,20),
                                                     cv=10,
                                                     n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean = np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color='blue',marker='o',markersize=5,label='traing accuracy')
plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')
plt.plot(train_sizes,test_mean,color='green',linestyle='--',marker='s',markersize=5,label='validation accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.0,1.0])
plt.show()






















