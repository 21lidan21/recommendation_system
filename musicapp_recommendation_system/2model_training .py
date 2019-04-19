
from surprise import Dataset
from surprise import evaluate, print_perf
from __future__ import (absolute_import, division, print_function, unicode_literals)
import os
import io

from surprise import KNNBaseline, Reader
from surprise import Dataset

import cPickle as pickle
# 默认载入movielens数据集
data = Dataset.load_builtin('ml-100k')
# k折交叉验证(k=3)
data.split(n_folds=3)
# 试一把SVD矩阵分解
algo = SVD()
# 在数据集上测试一下效果
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])
#输出结果
print_perf(perf)

# 指定文件所在路径
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')
# 告诉文本阅读器，文本的格式是怎么样的
reader = Reader(line_format='user item rating timestamp', sep='\t')
# 加载数据
data = Dataset.load_from_file(file_path, reader=reader)
# 手动切分成5折(方便交叉验证)
data.split(n_folds=5)

# 定义好需要优选的参数网格
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
# 使用网格搜索交叉验证
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
# 在数据集上找到最好的参数
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)
grid_search.evaluate(data)
# 输出调优的参数组 
# 输出最好的RMSE结果
print(grid_search.best_score['RMSE'])

# 输出对应最好的RMSE结果的参数
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}


# 重建歌单id到歌单名的映射字典
id_name_dic = pickle.load(open("popular_playlist.pkl","rb"))
print("加载歌单id到歌单名的映射字典完成...")
# 重建歌单名到歌单id的映射字典
name_id_dic = {}
for playlist_id in id_name_dic:
    name_id_dic[id_name_dic[playlist_id]] = playlist_id
print("加载歌单名到歌单id的映射字典完成...")


file_path = os.path.expanduser('./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 计算歌曲和歌曲之间的相似度
print("构建数据集...")
trainset = music_data.build_full_trainset()
#sim_options = {'name': 'pearson_baseline', 'user_based': False}

print("开始训练模型...")
#sim_options = {'user_based': False}
#algo = KNNBaseline(sim_options=sim_options)
algo = KNNBaseline()
algo.train(trainset)

current_playlist = name_id_dic.keys()[39]
print("歌单名称", current_playlist)

# 取出近邻
# 映射名字到id
playlist_id = name_id_dic[current_playlist]
print("歌单id", playlist_id)
# 取出来对应的内部user id => to_inner_uid
playlist_inner_id = algo.trainset.to_inner_uid(playlist_id)
print("内部id", playlist_inner_id)

playlist_neighbors = algo.get_neighbors(playlist_inner_id, k=10)

# 把歌曲id转成歌曲名字
# to_raw_uid映射回去
playlist_neighbors = (algo.trainset.to_raw_uid(inner_id)
                       for inner_id in playlist_neighbors)
playlist_neighbors = (id_name_dic[playlist_id]
                       for playlist_id in playlist_neighbors)

print()
print("和歌单 《", current_playlist, "》 最接近的10个歌单为：\n")
for playlist in playlist_neighbors:
    print(playlist, algo.trainset.to_inner_uid(name_id_dic[playlist]))

# 模板之针对用户进行预测
# 重建歌曲id到歌曲名的映射字典
song_id_name_dic = pickle.load(open("popular_song.pkl","rb"))
print("加载歌曲id到歌曲名的映射字典完成...")
# 重建歌曲名到歌曲id的映射字典
song_name_id_dic = {}
for song_id in song_id_name_dic:
    song_name_id_dic[song_id_name_dic[song_id]] = song_id
print("加载歌曲名到歌曲id的映射字典完成...")

#内部编码的4号用户
user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0], user_rating)
for song in items:
    print(algo.predict(user_inner_id, song, r_ui=1), song_id_name_dic[algo.trainset.to_raw_iid(song)])

surprise.dump.dump('./recommendation.model', algo=algo)
# 可以用下面的方式载入
algo = surprise.dump.load('./recommendation.model')
file_path = os.path.expanduser('./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 分成5折
music_data.split(n_folds=5)

### 模型评估
algo = KNNBaseline()
perf = evaluate(algo, music_data, measures=['RMSE', 'MAE'])

