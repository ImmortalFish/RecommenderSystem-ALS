# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:12:44 2017

@author: Administrator
"""
from pyspark import SparkContext,SparkConf
from pyspark.mllib.recommendation import ALS
from math import sqrt
from operator import add

conf = SparkConf().setAppName("MovieRecommendation").set("spark.executor.memory", "4g")
sc = SparkContext(conf=conf)
#为了保证迭代次数过多时不报错，具体原理不知道，搜了好久才看到这个解决方案
sc.setCheckpointDir("D:/WorkSpace/Spyder/checkpoint")


#处理用户评分数据
def HandleRating(line):
    user = line.strip().split(",")
    return int(user[3]) % 10,(int(user[0]),int(user[1]),float(user[2]))

#处理电影数据，返回[（电影id，电影名称）]
def HandleMovie(line):
    movie = line.strip().split(",")
    return int(movie[0]),movie[1]

#处理电影数据，返回[(电影id，(电影类型))],主要是为了解决根据电影类型推荐电影的冷启动问题
def HandleMovieList(line):
    movie = line.strip().split(",")
    return int(movie[0]),movie[2].split("|")

#读取用户0即需要推荐的用户的电影评分数据
def ReadMyRatingFile(path):
    file = open(path,'r')
    my_rating = [HandleRating(line)[1] for line in file]
    file.close()
    return my_rating

#读取新用户喜欢的电影类型数据，主要是为了解决冷启动问题
def ReadMyGenresFile(path):
    file = open(path)
    line = file.readline()
    my_movie_genres = line.strip().split(",")
    file.close()
    return my_movie_genres

#计算RMSE
def ComputeRMSE(model,data,n):
    prediction = model.predictAll(data.map(lambda x: (x[0],x[1])))
    prediction_rating = prediction.map(lambda x: ((x[0],x[1]),x[2])).join(data.map(lambda x: ((x[0],x[1]),x[2]))).values()
    return sqrt(prediction_rating.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

#冷启动的主函数
def ColdStart(path):
     #处理用户喜欢的电影的类型，返回一个包含这些类型的list
    my_genres = ReadMyGenresFile(path)
    #现在处理全部的电影数据，是为了获得电影id和类型的list
    movies_afterhandle = sc.textFile("D:/文档/毕业设计/数据集/ml-latest-small/movies.csv").map(HandleMovieList).collect()
    #这个list存储用户喜欢的电影类型与电影数据中匹配的电影id
    list_movie = []
    
    #开始匹配吧，把匹配成功的电影的id存入list_movie
    for m in range(len(movies_afterhandle)):
        if my_genres == movies_afterhandle[m][1]:
            list_movie.append(movies_afterhandle[m][0])
            
    #如果匹配的结果多于10个就选前10个推荐给用户
    if len(list_movie) > 10:
        print("根据您的喜好，为您推荐如下电影，希望您会喜欢：\n")
        for i in range(0,10):
            print(movie_data[list_movie[i]])
            
    #如果匹配的结果没有10个就全部推荐给用户
    else:
        print("根据您的喜好，为您推荐如下电影，希望您会喜欢：\n")
        for i in range(len(list_movie)):
            print(movie_data[list_movie[i]])  
    #如果没有匹配的结果的话，我还没系那个清楚，但是电影类型里面有一种没有类型的电影，打算推荐给用户，先问问老师吧
    sc.stop() 
    
#划分训练集
#训练集需要把要推荐用户的评分数据加进去
def DivideTrainSet(data_set,user_zero_data):
    return data_set.filter(lambda x: x[0] < 8).values().union(user_zero_data).repartition(4).cache()

#划分测试集
def DividTestSet(data_set):
    return data_set.filter(lambda x: x[0] >= 8).values().cache()

#统计数据集总数量
def CountDataSet(data_set):
    return data_set.count()

#用于统计用户数量
def CountUser(data_set):
    return data_set.values().map(lambda r: r[0]).distinct().count()

#用于统计电影数量
def CountMovie(data_set):
    return data_set.values().map(lambda r: r[1]).distinct().count()

#user_data是用来划分训练集和测试集的数据，包含很多条用户评分数据，具体多少后面会打印出来
user_data = sc.textFile("D:/文档/毕业设计/数据集/ml-latest-small/ratings.csv").map(HandleRating)
#movie_data是电影的数据，具体多少个电影后面也会打印出来，这里将其弄成dict主要是为了后面出推荐结果时好寻找并打印
movie_data = dict(sc.textFile("D:/文档/毕业设计/数据集/ml-latest-small/movies.csv").map(HandleMovie).collect())

#推荐函数
def Recommendation(path):
    #这是在读取用户的评分数据
    my_ratings = ReadMyRatingFile(path)
    my_data_rdd = sc.parallelize(my_ratings,1)
    
    #统计数据集的用户、电影、评分数量
    num_rating = CountDataSet(user_data)
    num_user = CountUser(user_data)
    num_movie = CountMovie(user_data)
    print("数据集总共有%d条评分数据." % num_rating)
    print("数据集共有%d个用户对%d部电影进行了评分." % (num_user,num_movie))
        
    #下面是划分训练集和测试集，比例是0.8：0.2
    train_data = DivideTrainSet(user_data,my_data_rdd)
    test_data = DividTestSet(user_data)
    
    #计算训练集和测试集的数量
    num_train = CountDataSet(train_data)
    num_test = CountDataSet(test_data)
    print("训练集有%d条数据，测试集有%d条数据" % (num_train,num_test))
    
    #参数设置
    ranks = 20
    lambdas = 0.01
    iterations = 40
    blocks = -1
    alphas = 0.01
    
    #这句代码也是为了防止由于迭代次数过多而产生错误
    ALS.checkpointInterval = 2
        
    #开始训练模型
    model = ALS.trainImplicit(train_data,ranks,iterations,lambdas,blocks,alphas)
        
    #在测试集上计算RMSE
    test_RMSE = ComputeRMSE(model,test_data,num_test)
    print("在测试集上的RMSE：%f" % test_RMSE)
    
    #进行个性化推荐
    #找出用户已经看过的电影ID
    my_rated_movie_ids = set([x[1] for x in my_ratings])
    
    #对比电影数据集，找出那些用户没有看过的电影
    my_not_rate_movie = sc.parallelize([m for m in movie_data if m not in my_rated_movie_ids])

    #开始推荐
    predictions = model.predictAll(my_not_rate_movie.map(lambda x: (0,x))).collect()
    reccommendation = sorted(predictions,key = lambda x: x[2],reverse = True)[:20]
    
    #打印推荐结果
    prediction_movieid = []
    print("######################################################")
    print("######################################################")
    print("######################################################")
    print("为您推荐以下电影：")
    for i in range(len(reccommendation)):
        prediction_movieid.append(reccommendation[i][1])
        print("%2d: %d  %s" % (i + 1,reccommendation[i][1],movie_data[reccommendation[i][1]]))
        
    #后面这部分是为了测试推荐的结果
    #把需要推荐的用户的评分数据截取一部分出来与推荐结果进行对比，看一看精度
    rating_compare = ReadMyRatingFile("D:/文档/毕业设计/数据集/ml-latest-small/test_2_last.data")
    rating_compare_rdd = sc.parallelize(rating_compare,1)
    rating_compare_movieid = rating_compare_rdd.map(lambda x: x[1]).collect()
    
    print("######################################################")
    print(prediction_movieid)
    print("######################################################")
    print(rating_compare_movieid)
    print("######################################################")
    
    print("看是否有相同的电影：")
    compare = [l for l in prediction_movieid if l in rating_compare_movieid]
    if(len(compare) == 0):
        print("磊哥啊，再改进改进吧，一定可以的")
    else:
        for m in compare:
            print("预测中的电影ID：%d  电影名称：%s" % (m,movie_data[m]))   
    sc.stop() 


#主函数
def main():

    #下面这个条件判断是推荐的主体
    #首先询问用户是否有有观影数据，即之前是否看过这些电影，如果是则输入yes
    #如果用户输入的是no，就是一个冷启动的问题，系统没有任何用户的数据，就只有引导用户进行选择、
    #让用户将喜欢的电影类型写到一个文件中，然后对电影数据中的类型进行全字匹配，把这个匹配的电影推荐给用户
    #！！！这个冷启动的方案比较简单，但是目前我能想到的就是这个了，之后如果有时间再改吧，先打个标签在这里！！！！！！！
    if_see_movie = input("您以前是否有过观影记录？如果有请输入yes，否则输入no，退出请输入quit：\n")
    
    if(if_see_movie == "quit"):
        print("\n感谢您的使用，再见！")
        sc.stop()
    
    #这个冷启动的解决方法，按照用户喜欢的电影类型为其推荐电影    
    elif(if_see_movie == "no"):
        print("\n请在txt文件中输入你喜欢看的电影类型,以逗号隔开,有如下类型：")
        print("Action,Adventure,Animation,Children,Comedy,Crime,Documentary,Drama,Fantasy")
        print("Film-Noir,Horror,Musical,Mystery,Romance,Sci-Fi,Thriller,War,Western")
        my_genres_data_path = input("请输入该txt文件的绝对路径：")
        print("请稍等，系统正在根据您的喜好为您推荐电影...\n")
        ColdStart(my_genres_data_path)
        
    #这个呢就是有用户的电影评分数据的推荐
    #先将用户的评分数据加入到训练集中进行训练模型
    #再给用户推荐50部电影
    elif(if_see_movie == "yes"):
        print("\n请将您以前的观影记录按如下格式记录在.data文件中")
        print("格式为：0,电影Id,您对电影的评分(满分为5),时间")
        my_rating_data_path = input("请输入该文件的绝对路径：")
        print("请稍等，系统正在为您推荐电影...\n")
        Recommendation(my_rating_data_path)
      
    else:
        print("您是不是输入有误，请仔细检查并阅读上面的说明！")
        main()

if __name__ == "__main__":
      print("\n******************欢迎使用电影推荐原型系统******************")
      main() 