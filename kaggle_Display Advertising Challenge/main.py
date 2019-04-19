import os
import sys
import click
import random
import collections

import numpy as np
import lightgbm as lgb
import json
import pandas as pd
from sklearn.metrics import mean_squared_error


import pickle

def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params.p', mode='rb'))


def save_params_with_name(params, name):
    """
    Save parameters to file
    """
    pickle.dump(params, open('{}.p'.format(name), 'wb'))


def load_params_with_name(name):
    """
    Load parameters from file
    """
    return pickle.load(open('{}.p'.format(name), mode='rb'))

# There are 13 integer features and 26 categorical features
continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]

class ContinuousFeatureGenerator:
    """
    Normalize the integer features to [0, 1] by min-max normalization
    """
    def _int_(self,num_feature):
        self.num_feature=num_feature
        self.min=[sys.maxsize]*num_feature
        self.max=[-sys.maxsize]*num_feature

    def build(self,datafile,continous_features):
        with open(datafile,'r') as f:
            for line in f：
            features=line.rstrip('\n').split('\t')
            for i in range(0,self.num_feature):
                val=features[continous_features[i]]
                if val!='':
                    val=int(val)
                    if val>continous_clip[i]:
                        val=continous_clip[i]
                    self.min[i]=min(self.min[i],val)
                    self.max[i]=max(self.max[i],val)
    def gen(self,idx,val):
        if val='':
            return 0.0
        val =float(val)
        return (val-self.min[idx])/(self.max[idx]-self.min[idx])

class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """
    def __init__(self, num_feature):
        self.dicts=[]
        self.num_feature=num_feature
        for i in range(0,num_feature):
            self.dicts.append(collections.default(int))
    def build(self,datafile,categorial_features,cutoff=0):
        with open(datafile,'r') as f:
            for line in f:
                features=line.rstrip('\n').split('\t')
                for i in range(0,self.num_feature):
                    if features[categorial_features[i]]!='':
                        self.dicts[i][features[categorial_features[i]]]+=1
        for i in range(0,self.num_feature):
            self.dicts[i]=filter(lambda x:x[1]>=cutoff,
                                self.dicts[i].items())
            self.dicts[i]=sorted(self.dicts[i],key=lambda x:(-x[1],x[0]))
            vocabs,_=list(zip(*self.dicts[i]))
            self.dicts[i]=dict(zip(vocabs,range(1,len(vocabs)+1)))
            self.dicts[i]['<unk>'] =0               
    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))

def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.

    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
        """
    dists=ContinuousFeatureGenerator(len(continous_features))
    dists.build(os.path.join(datadir,'./data/dac_sample/dac_sample.txt'),continous_features)

    dicts=CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir,'./data/dac_sample/dac_sample.txt'),categorial_features,cutoff=200)
    dicts_sizes=dicts.dicts_sizes()
    categorial_feature_offset=[0]
    for i in range(1,len(categorial_features)):
        offset=categorial_feature_offset[i-1]+dict_sizes[i-1]
        categorial_feature_offset.append(offset)
    random.seed(0)

    # 90% of the data are used for training, and 10% of the data are used
    # for validation.
    train_ffm = open(os.path.join(outdir, 'train_ffm.txt'), 'w')
    valid_ffm = open(os.path.join(outdir, 'valid_ffm.txt'), 'w')

    train_lgb = open(os.path.join(outdir, 'train_lgb.txt'), 'w')
    valid_lgb = open(os.path.join(outdir, 'valid_lgb.txt'), 'w')

     with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid.txt'), 'w') as out_valid:
            with open(os.path.join(datadir, 'train.txt'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')
                    continous_feats = []
                    continous_vals = []
                    for i in range(0, len(continous_features)):

                        val = dists.gen(i, features[continous_features[i]])
                        continous_vals.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                        continous_feats.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))#('{0}'.format(val))

                    categorial_vals = []
                    categorial_lgb_vals = []
                    for i in range(0, len(categorial_features)):
                        val = dicts.gen(i, features[categorial_features[i]]) + categorial_feature_offset[i]
                        categorial_vals.append(str(val))
                        val_lgb = dicts.gen(i, features[categorial_features[i]])
                        categorial_lgb_vals.append(str(val_lgb))

                    continous_vals = ','.join(continous_vals)
                    categorial_vals = ','.join(categorial_vals)
                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        train_ffm.write('\t'.join(label) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                        train_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                        
                        train_lgb.write('\t'.join(label) + '\t')
                        train_lgb.write('\t'.join(continous_feats) + '\t')
                        train_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

                    else:
                        out_valid.write(','.join(
                            [continous_vals, categorial_vals, label]) + '\n')
                        valid_ffm.write('\t'.join(label) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                        valid_ffm.write('\t'.join(
                            ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                                                
                        valid_lgb.write('\t'.join(label) + '\t')
                        valid_lgb.write('\t'.join(continous_feats) + '\t')
                        valid_lgb.write('\t'.join(categorial_lgb_vals) + '\n')
                        
    train_ffm.close()
    valid_ffm.close()

    train_lgb.close()
    valid_lgb.close()

    test_ffm = open(os.path.join(outdir, 'test_ffm.txt'), 'w')
    test_lgb = open(os.path.join(outdir, 'test_lgb.txt'), 'w')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_feats = []
                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append(
                        "{0:.6f}".format(val).rstrip('0').rstrip('.'))
                    continous_feats.append(
                            "{0:.6f}".format(val).rstrip('0').rstrip('.'))#('{0}'.format(val))

                categorial_vals = []
                categorial_lgb_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i,
                                    features[categorial_features[i] -
                                             1]) + categorial_feature_offset[i]
                    categorial_vals.append(str(val))

                    val_lgb = dicts.gen(i, features[categorial_features[i] - 1])
                    categorial_lgb_vals.append(str(val_lgb))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)

                out.write(','.join([continous_vals, categorial_vals]) + '\n')
                
                test_ffm.write('\t'.join(['{}:{}:{}'.format(ii, ii, val) for ii,val in enumerate(continous_vals.split(','))]) + '\t')
                test_ffm.write('\t'.join(
                    ['{}:{}:1'.format(ii + 13, str(np.int32(val) + 13)) for ii, val in enumerate(categorial_vals.split(','))]) + '\n')
                                                                
                test_lgb.write('\t'.join(continous_feats) + '\t')
                test_lgb.write('\t'.join(categorial_lgb_vals) + '\n')

    test_ffm.close()
    test_lgb.close()
    return dict_sizes

dict_sizes = preprocess('./data/raw','./data')
save_params_with_name((dict_sizes), 'dict_sizes') #pickle.dump((dict_sizes), open('dict_sizes.p', 'wb'))
dict_sizes = load_params_with_name('dict_sizes') #pickle.load(open('dict_sizes.p', mode='rb'))
sum(dict_sizes)


# 数据准备好了，开始调用LibFFM，训练FFM模型。
# learning rate是0.1，迭代32次，训练好后保存的模型文件是model_ffm。

import subprocess, sys, os, time
NR_THREAD = 1

cmd = './libffm/libffm/ffm-train --auto-stop -r 0.1 -t 32 -s {nr_thread} -p ./data/valid_ffm.txt ./data/train_ffm.txt model_ffm'.format(nr_thread=NR_THREAD) 
os.popen(cmd).readlines()
# FFM模型训练好了，我们把训练、验证和测试数据输入给FFM，得到FFM层的输出，输出的文件名为*.out.logit
cmd = './libffm/libffm/ffm-predict ./data/train_ffm.txt model_ffm tr_ffm.out'.format(nr_thread=NR_THREAD) 
os.popen(cmd).readlines()
cmd = './libffm/libffm/ffm-predict ./data/valid_ffm.txt model_ffm va_ffm.out'.format(nr_thread=NR_THREAD) 
os.popen(cmd).readlines()
cmd = './libffm/libffm/ffm-predict ./data/test_ffm.txt model_ffm te_ffm.out true'.format(nr_thread=NR_THREAD) 
os.popen(cmd).readlines()

# 现在调用LightGBM训练GBDT模型，因为决策树较容易过拟合，我们设置树的个数为32，叶子节点数设为30，深度就不设置了，学习率设为0.05。
def lgb_pred(tr_path, va_path, _sep = '\t', iter_num = 32):
    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv(tr_path, header=None, sep=_sep)
    df_test = pd.read_csv(va_path, header=None, sep=_sep)
    
    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values
    
    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    
    # specify your configurations as a dict
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'auc', 'logloss'},
        'num_leaves': 30,
#         'max_depth': 7,
        'num_trees': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    
    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=iter_num,
                    valid_sets=lgb_eval,
                    feature_name=["I1","I2","I3","I4","I5","I6","I7","I8","I9","I10","I11","I12","I13","C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"],
                    categorical_feature=["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26"],
                    early_stopping_rounds=5)
    
    print('Save model...')
    # save model to file
    gbm.save_model('lgb_model.txt')
    
    print('Start predicting...')
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

    return gbm,y_pred,X_train,y_train
    
gbm,y_pred,X_train ,y_train = lgb_pred('./data/train_lgb.txt', './data/valid_lgb.txt', '\t', 256)

# 查看每个特征的重要程度
gbm.feature_importance()
gbm.feature_importance("gain")
# 我们把每个特征的重要程度排个序看看
def ret_feat_impt(gbm):
    gain = gbm.feature_importance("gain").reshape(-1, 1) / sum(gbm.feature_importance("gain"))
    col = np.array(gbm.feature_name()).reshape(-1, 1)
    return sorted(np.column_stack((col, gain)),key=lambda x: x[1],reverse=True)
ret_feat_impt(gbm)

dump = gbm.dump_model()
save_params_with_name((gbm, dump), 'gbm_dump') 
gbm, dump = load_params_with_name('gbm_dump') 

# 用LightGBM的输出生成FM数据
def generat_lgb2fm_data(outdir, gbm, dump, tr_path, va_path, te_path, _sep = '\t'):
    with open(os.path.join(outdir, 'train_lgb2fm.txt'), 'w') as out_train:
        with open(os.path.join(outdir, 'valid_lgb2fm.txt'), 'w') as out_valid:
            with open(os.path.join(outdir, 'test_lgb2fm.txt'), 'w') as out_test:
                df_train_ = pd.read_csv(tr_path, header=None, sep=_sep)
                df_valid_ = pd.read_csv(va_path, header=None, sep=_sep)
                df_test_= pd.read_csv(te_path, header=None, sep=_sep)

                y_train_ = df_train_[0].values
                y_valid_ = df_valid_[0].values                

                X_train_ = df_train_.drop(0, axis=1).values
                X_valid_ = df_valid_.drop(0, axis=1).values
                X_test_= df_test_.values
   
                train_leaves= gbm.predict(X_train_, num_iteration=gbm.best_iteration, pred_leaf=True)
                valid_leaves= gbm.predict(X_valid_, num_iteration=gbm.best_iteration, pred_leaf=True)
                test_leaves= gbm.predict(X_test_, num_iteration=gbm.best_iteration, pred_leaf=True)

                tree_info = dump['tree_info']
                tree_counts = len(tree_info)
                for i in range(tree_counts):
                    train_leaves[:, i] = train_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
                    valid_leaves[:, i] = valid_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
                    test_leaves[:, i] = test_leaves[:, i] + tree_info[i]['num_leaves'] * i + 1
#                     print(train_leaves[:, i])
#                     print(tree_info[i]['num_leaves'])

                for idx in range(len(y_train_)):            
                    out_train.write((str(y_train_[idx]) + '\t'))
                    out_train.write('\t'.join(
                        ['{}:{}'.format(ii, val) for ii,val in enumerate(train_leaves[idx]) if float(val) != 0 ]) + '\n')
                    
                for idx in range(len(y_valid_)):                   
                    out_valid.write((str(y_valid_[idx]) + '\t'))
                    out_valid.write('\t'.join(
                        ['{}:{}'.format(ii, val) for ii,val in enumerate(valid_leaves[idx]) if float(val) != 0 ]) + '\n')
                    
                for idx in range(len(X_test_)):                   
                    out_test.write('\t'.join(
                        ['{}:{}'.format(ii, val) for ii,val in enumerate(test_leaves[idx]) if float(val) != 0 ]) + '\n')
generat_lgb2fm_data('./data', gbm, dump, './data/train_lgb.txt', './data/valid_lgb.txt', './data/test_lgb.txt', '\t')

# 训练FM调用LibFM进行训练。
# 迭代64次，使用sgd训练，学习率是0.00000001，训练好的模型保存为文件fm_model。
# 训练输出的log，Train和Test的数值不是loss，是accuracy。

cmd = './libfm/libfm/bin/libFM -task c -train ./data/train_lgb2fm.txt -test ./data/valid_lgb2fm.txt -dim ’1,1,8’ -iter 64 -method sgd -learn_rate 0.00000001 -regular ’0,0,0.01’ -init_stdev 0.1 -save_model fm_model'
os.popen(cmd).readlines()

# FM模型训练好了，我们把训练、验证和测试数据输入给FM，得到FM层的输出，输出的文件名为*.fm.logits
cmd = './libfm/libfm/bin/libFM -task c -train ./data/train_lgb2fm.txt -test ./data/valid_lgb2fm.txt -dim ’1,1,8’ -iter 32 -method sgd -learn_rate 0.00000001 -regular ’0,0,0.01’ -init_stdev 0.1 -load_model fm_model -train_off true -prefix tr'
os.popen(cmd).readlines()

cmd = './libfm/libfm/bin/libFM -task c -train ./data/valid_lgb2fm.txt -test ./data/valid_lgb2fm.txt -dim ’1,1,8’ -iter 32 -method sgd -learn_rate 0.00000001 -regular ’0,0,0.01’ -init_stdev 0.1 -load_model fm_model -train_off true -prefix va'
os.popen(cmd).readlines()

cmd = './libfm/libfm/bin/libFM -task c -train ./data/test_lgb2fm.txt -test ./data/valid_lgb2fm.txt -dim ’1,1,8’ -iter 32 -method sgd -learn_rate 0.00000001 -regular ’0,0,0.01’ -init_stdev 0.1 -load_model fm_model -train_off true -prefix te -test2predict true'
os.popen(cmd).readlines()

# 开始构建模型
embed_dim = 32
sparse_max = 30000 # sparse_feature_dim = 117568
sparse_dim = 26
dense_dim = 13
out_dim = 400
# 全数据迭代器
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]
# 下采样数据迭代器
def get_batches_downsample(Xs, ys, batch_size):
    ind_0 = ys==0
    ind_1 = ys==1
    Xs_0 = Xs[ind_0]
    ys_0 = ys[ind_0]
    Xs_1 = Xs[ind_1]
    ys_1 = ys[ind_1]
    sampling_ind = np.random.permutation(Xs_0.shape[0])[:Xs_1.shape[0]]
    Xs_0_sampling = Xs_0[sampling_ind]
    ys_0_sampling = ys_0[sampling_ind]
    Xs_downsampled = np.concatenate((Xs_0_sampling, Xs_1))
    ys_downsampled = np.concatenate((ys_0_sampling, ys_1))
    downsampled_ind = np.random.permutation(Xs_downsampled.shape[0])
    Xs_downsampled = Xs_downsampled[downsampled_ind]
    ys_downsampled = ys_downsampled[downsampled_ind]
    for start in range(0, len(Xs_downsampled), batch_size):
        end = min(start + batch_size, len(Xs_downsampled))
        yield Xs_downsampled[start:end], ys_downsampled[start:end]

# 构建计算图

# 如前所述，将FFM和FM层的输出经过全连接层，再和数值特征、嵌入向量的三层全连接层的输出连接在一起，做Logistic回归。

# 采用LogLoss损失，FtrlOptimizer优化损失。

import tensorflow as tf
import datetime
from tensorflow import keras
from tensorflow.python.ops import summary_ops_v2
import time
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
import datetime
from sklearn.metrics import log_loss
# from sklearn.learning_curve import learning_curve
from sklearn.model_selection import learning_curve
from sklearn import metrics as sk_metrics

MODEL_DIR = "./models"


class ctr_network(object):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.best_loss = 9999

        self.losses = {'train': [], 'test': []}
        self.pred_lst = []
        self.test_y_lst = []
        
        # 定义输入
        dense_input = tf.keras.layers.Input(shape=(dense_dim,), name='dense_input')
        sparse_input = tf.keras.layers.Input(shape=(sparse_dim,), name='sparse_input')
        FFM_input = tf.keras.layers.Input(shape=(1,), name='FFM_input')
        FM_input = tf.keras.layers.Input(shape=(1,), name='FM_input')

        # 输入类别特征，从嵌入层获得嵌入向量
        sparse_embed_layer = tf.keras.layers.Embedding(sparse_max, embed_dim, input_length=sparse_dim)(sparse_input)
        sparse_embed_layer = tf.keras.layers.Reshape([sparse_dim * embed_dim])(sparse_embed_layer)

        # 输入数值特征，和嵌入向量链接在一起经过三层全连接层
        input_combine_layer = tf.keras.layers.concatenate([dense_input, sparse_embed_layer])  # (?, 845 = 832 + 13)
        fc1_layer = tf.keras.layers.Dense(out_dim, name="fc1_layer", activation='relu')(input_combine_layer)
        fc2_layer = tf.keras.layers.Dense(out_dim, name="fc2_layer", activation='relu')(fc1_layer)
        fc3_layer = tf.keras.layers.Dense(out_dim, name="fc3_layer", activation='relu')(fc2_layer)

        ffm_fc_layer = tf.keras.layers.Dense(1, name="ffm_fc_layer")(FFM_input)
        fm_fc_layer = tf.keras.layers.Dense(1, name="fm_fc_layer")(FM_input)
        feature_combine_layer = tf.keras.layers.concatenate([ffm_fc_layer, fm_fc_layer, fc3_layer], 1)  # (?, 402)

        logits_output = tf.keras.layers.Dense(1, name="logits_layer", activation='sigmoid')(feature_combine_layer)

        self.model = tf.keras.Model(inputs=[dense_input, sparse_input, FFM_input, FM_input], outputs=[logits_output])
        self.model.summary()

        self.optimizer = tf.compat.v1.train.FtrlOptimizer(0.01)  # tf.keras.optimizers.Adam(0.01)
        self.ComputeLoss = tf.keras.losses.LogLoss()

        if tf.io.gfile.exists(MODEL_DIR):
            #             print('Removing existing model dir: {}'.format(MODEL_DIR))
            #             tf.io.gfile.rmtree(MODEL_DIR)
            pass
        else:
            tf.io.gfile.makedirs(MODEL_DIR)

        train_dir = os.path.join(MODEL_DIR, 'summaries', 'train')
        test_dir = os.path.join(MODEL_DIR, 'summaries', 'eval')

#         self.train_summary_writer = summary_ops_v2.create_file_writer(train_dir, flush_millis=10000)
#         self.test_summary_writer = summary_ops_v2.create_file_writer(test_dir, flush_millis=10000, name='test')

        checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
        self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

        # Restore variables on creation if a checkpoint exists.
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    def compute_metrics(self, labels, pred):
        correct_prediction = tf.equal(tf.keras.backend.cast(pred > 0.5, 'float32'), labels)
        accuracy = tf.reduce_mean(tf.keras.backend.cast(correct_prediction, 'float32'), name="accuracy")
        return accuracy  

    @tf.function
    def train_step(self, x, y):
        # Record the operations used to compute the loss, so that the gradient
        # of the loss with respect to the variables can be computed.
        metrics = 0
        with tf.GradientTape() as tape:
            pred = self.model([x[0],
                               x[1],
                               x[2],
                               x[3]], training=True)
            loss = self.ComputeLoss(y, pred)
            metrics = self.compute_metrics(y, pred)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics, pred

    def training(self, train_dataset, test_dataset, downsample_flg=True, epochs=1, log_freq=50):

        train_X, train_y = train_dataset

        for epoch_i in range(epochs):
            if downsample_flg:
                train_batches = get_batches_downsample(train_X, train_y, self.batch_size)
                batch_num = (len(train_y[train_y==1])*2 // self.batch_size)
            else:
                train_batches = get_batches(train_X, train_y, self.batch_size)
                batch_num = len(train_X) // self.batch_size

            train_start = time.time()
#             with self.train_summary_writer.as_default():
            if True:
                start = time.time()
                # Metrics are stateful. They accumulate values and return a cumulative
                # result when you call .result(). Clear accumulated values with .reset_states()
                avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
                avg_acc = tf.keras.metrics.Mean('acc', dtype=tf.float32)
                avg_auc = tf.keras.metrics.Mean('auc', dtype=tf.float32)

                # Datasets can be iterated over like any other Python iterable.
                for batch_i in range(batch_num):
                    x, y = next(train_batches)
                    if len(x) < self.batch_size:
                        break
                    
                    loss, metrics, pred = self.train_step([x.take([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 1),
                               x.take(
                                   [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                    36, 37, 38, 39, 40], 1),
                               np.reshape(x.take(0, 1), [self.batch_size, 1]),
                               np.reshape(x.take(1, 1), [self.batch_size, 1])], np.expand_dims(y, 1))
                    avg_loss(loss)
                    avg_acc(metrics)

                    prediction = tf.reshape(pred, y.shape)
                    self.losses['train'].append(loss)

                    if (np.mean(y) != 0):
                        auc = sk_metrics.roc_auc_score(y, prediction)
                    else:
                        auc = -1

                    avg_auc(auc)
                    if tf.equal((epoch_i * (batch_num) + batch_i) % log_freq, 0):
#                         summary_ops_v2.scalar('loss', avg_loss.result(), step=self.optimizer.iterations)
                        #                         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=self.optimizer.iterations)
#                         summary_ops_v2.scalar('acc', avg_acc.result(), step=self.optimizer.iterations)

                        rate = log_freq / (time.time() - start)
                        
                        print('Epoch {:>3} Batch {:>4}/{} Loss: {:0.6f} acc: {:0.6f} auc = {} ({} steps/sec)'.format(
                            epoch_i, batch_i, batch_num, avg_loss.result(), (avg_acc.result()), avg_auc.result(), rate))

                        avg_auc.reset_states()
                        avg_loss.reset_states()
                        
                        avg_acc.reset_states()
                        start = time.time()

            train_end = time.time()
            print('\nTrain time for epoch #{} : {}'.format(epoch_i + 1, train_end - train_start))
#             with self.test_summary_writer.as_default():
            self.testing(test_dataset)
            # self.checkpoint.save(self.checkpoint_prefix)
        self.export_path = os.path.join(MODEL_DIR, 'export')
        tf.saved_model.save(self.model, self.export_path)

    def testing(self, test_dataset):
        test_X, test_y = test_dataset
        test_batches = get_batches(test_X, test_y, self.batch_size)

        """Perform an evaluation of `model` on the examples from `dataset`."""
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_acc = tf.keras.metrics.Mean('acc', dtype=tf.float32)
        avg_auc = tf.keras.metrics.Mean('auc', dtype=tf.float32)
        avg_prediction = tf.keras.metrics.Mean('prediction', dtype=tf.float32)

        self.pred_lst=[]
        self.test_y_lst=[]
        
        batch_num = (len(test_X) // self.batch_size)
        for batch_i in range(batch_num):
            x, y = next(test_batches)
            if len(x) < self.batch_size:
                break
            
            pred = self.model([x.take([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], 1),
                               x.take(
                                   [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                                    36, 37, 38, 39, 40], 1),
                               np.reshape(x.take(0, 1), [self.batch_size, 1]),
                               np.reshape(x.take(1, 1), [self.batch_size, 1])], training=False)
            test_loss = self.ComputeLoss(np.expand_dims(y, 1), pred)
            avg_loss(test_loss)
            acc = self.compute_metrics(np.expand_dims(y, 1), pred)
            avg_acc(acc)

            # 保存测试损失和准确率
            prediction = tf.reshape(pred, y.shape)
            avg_prediction(prediction)
            self.losses['test'].append(test_loss)

            self.pred_lst.append(prediction)
            self.test_y_lst.append(y)

            if (np.mean(y) != 0):
                auc = sk_metrics.roc_auc_score(y, prediction)
            else:
                auc = -1
            avg_auc(auc)

        self.pred_lst = np.concatenate([val for val in self.pred_lst])
        self.test_y_lst = np.concatenate([val for val in self.test_y_lst])
        print('Model test set loss: {:0.6f}  acc: {:0.6f}  auc = {} prediction = {}'.format(
            avg_loss.result(), avg_acc.result(), avg_auc.result(), avg_prediction.result()))
        print(sk_metrics.classification_report(self.test_y_lst, tf.keras.backend.cast((self.pred_lst) > 0.5, 'float32')))
#         summary_ops_v2.scalar('loss', avg_loss.result(), step=step_num)
        #         summary_ops_v2.scalar('mae', self.ComputeMetrics.result(), step=step_num)
#         summary_ops_v2.scalar('acc', avg_acc.result(), step=step_num)

        if avg_loss.result() < self.best_loss:
            self.best_loss = avg_loss.result()
            print("best loss = {}".format(self.best_loss))
            self.checkpoint.save(self.checkpoint_prefix)

    def predict_click(self, x, axis = 0):
        clicked = self.model([np.reshape(x.take([2,3,4,5,6,7,8,9,10,11,12,13,14],axis), [1 if axis == 0 else len(x.take([2,3,4,5,6,7,8,9,10,11,12,13,14],axis)), 13]),
                               np.reshape(x.take([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],axis), [1 if axis == 0 else len(x.take([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],axis)), 26]),
                               np.reshape(x.take(0,axis), [1 if axis == 0 else len(x.take(0,axis)), 1]),
                               np.reshape(x.take(1,axis), [1 if axis == 0 else len(x.take(0,axis)), 1])])

        return (np.int32(np.array(clicked) > 0.5))
# 超参
# Number of Epochs
num_epochs = 1
# Batch Size
batch_size = 32

# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 25

save_dir = './save'

ffm_tr_out_path = './tr_ffm.out.logit'
ffm_va_out_path = './va_ffm.out.logit'
fm_tr_out_path = './tr.fm.logits'
fm_va_out_path = './va.fm.logits'
train_path = './data/train.txt'
valid_path = './data/valid.txt'

# 读取FFM的输出
ffm_train = pd.read_csv(ffm_tr_out_path, header=None)    
ffm_train = ffm_train[0].values

ffm_valid = pd.read_csv(ffm_va_out_path, header=None)    
ffm_valid = ffm_valid[0].values
# 读取FM的输出
fm_train = pd.read_csv(fm_tr_out_path, header=None)    
fm_train = fm_train[0].values

fm_valid = pd.read_csv(fm_va_out_path, header=None)    
fm_valid = fm_valid[0].values

# 读取数据集
# 将DNN数据和FM、FFM的输出数据读取出来，并连接在一起
train_data = pd.read_csv(train_path, header=None)    
train_data = train_data.values

valid_data = pd.read_csv(valid_path, header=None)    
valid_data = valid_data.values

cc_train = np.concatenate((ffm_train.reshape(-1, 1), fm_train.reshape(-1, 1), train_data), 1)
cc_valid = np.concatenate((ffm_valid.reshape(-1, 1), fm_valid.reshape(-1, 1), valid_data), 1)

np.random.shuffle(cc_train)
np.random.shuffle(cc_valid)

train_y = cc_train[:,-1].astype(np.float32)
test_y = cc_valid[:,-1].astype(np.float32)

train_X = cc_train[:,0:-1].astype(np.float32)
test_X = cc_valid[:,0:-1].astype(np.float32)

# 来看训练数据和验证数据的平均点击率
np.mean(train_y)
np.mean(test_y)
# 训练网络
ctr_net=ctr_network()
ctr_net.training((train_X, train_y), (test_X, test_y), True, 1)

# 全数据的训练
ctr_net=ctr_network()
ctr_net.training((train_X, train_y), (test_X, test_y), False, 1)




# 查看loss曲线
# 下采样的loss
plt.plot(ctr_net.losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()

plt.plot(ctr_net.losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()
# 全数据的loss
plt.plot(ctr_net.losses['train'], label='Training loss')
plt.legend()
_ = plt.ylim()

plt.plot(ctr_net.losses['test'], label='Test loss')
plt.legend()
_ = plt.ylim()

# 使用测试数据进行测试
ffm_test_out_path = './te_ffm.out.logit'
fm_test_out_path = './te.fm.logits'
test_path = './data/test.txt'

ffm_test = pd.read_csv(ffm_test_out_path, header=None)    
ffm_test = ffm_test[0].values

fm_test = pd.read_csv(fm_test_out_path, header=None)    
fm_test = fm_test[0].values

test_data = pd.read_csv(test_path, header=None)    
test_data = test_data.values

pred_test_X = np.concatenate((ffm_test.reshape(-1, 1), fm_test.reshape(-1, 1), test_data), 1).astype(np.float32)

# pred_test_X = cc_test[:,0:-1]

save_params_with_name((pred_test_X), "pred_test_X")
pred_test_X = load_params_with_name("pred_test_X")
len(pred_test_X)
ctr_net=ctr_network(1)

# 下采样训练的网络预测
ctr_net.predict_click(pred_test_X[0])
ctr_net.predict_click(pred_test_X[:20], 1)
# 全数据训练的网络预测
ctr_net.predict_click(pred_test_X[0])
ctr_net.predict_click(pred_test_X[:20], 1)

def get_test_batches(Xs, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end]

def predict_test(ctr_net, batch_size, axis = 1):
    if True:
        test_batches = get_test_batches(pred_test_X, batch_size)
        total_num = len(pred_test_X)
        
        pred_lst = []
        for batch_i in range(total_num // batch_size):
            x = next(test_batches)
            # Get Prediction
            clicked = ctr_net.predict_click(x, axis)
            pred_lst.append(clicked)
            if ((total_num // batch_size) + batch_i) % show_every_n_batches == 0:
                        print('Batch {:>4}/{}   mean click = {}'.format(
                            batch_i,
                            (total_num // batch_size),
                            np.mean(np.array(clicked))))
#         print(np.array(clicked))
        return pred_lst

# 下采样训练的网络预测
pred_lst = predict_test(ctr_net, 64)
np.mean(pred_lst)

# 全数据训练的网络预测¶
pred_lst = predict_test(ctr_net, 64)
np.mean(pred_lst)
















