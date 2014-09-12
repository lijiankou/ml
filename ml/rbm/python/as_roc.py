import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def LoadPPM(path):
    line = open(path, 'rb')
    line.readline()
    dic = {}
    i = 0
    for l in line:
        t = l.split()
        dic[i] = {}
        for j in range(len(t)):
            if float(t[j]) > 0:
                dic[i][j] = float(t[j])
        i += 1
    line.close()
    return dic

def LoadTest(path):
    line = open(path, 'rb')
    rating = []
    for l in line:
        t = l.split()
        for i in range(len(t)):
            t[i] = int(t[i])
        rating.append(t)
    line.close()
    return rating

def RatingImputationGROC(rating, p_p_m):
    predict = []
    real = []
    for r in rating:
        if r[1] in p_p_m[r[0]]: 
            predict.append(p_p_m[r[0]][r[1]])
            real.append(r[2])
    return predict, real

#movie user rating
def LoadMovieUserDicForImplicitROC(path):
    lines = open(path)
    movie = {}
    for l in lines:
        t = l.split()
        m_id = int(t[1])
        if m_id not in movie:
            movie[m_id] = {}
        movie[m_id][int(t[0])] = int(t[2])
    lines.close()
    return movie

def ImplicitGROC(movie_user_dic, p_p_m):
    predict = []
    real = []
    print len(p_p_m)
    for m in movie_user_dic:
        for id in range(1, 944):
            if id not in p_p_m or len(p_p_m[id]) == 0 or m not in p_p_m[id]:
                predict.append(0)
            else:
                predict.append(p_p_m[id][m])
            if id in movie_user_dic[m]:
                real.append(2)
            else:
                real.append(1)
    return predict, real

def RatingPredictionGROC(movie_user_dic, p_p_m):
    predict = []
    real = []
    print len(p_p_m)
    for m in movie_user_dic:
        for id in range(1, 944):
            if id not in p_p_m or len(p_p_m[id]) == 0 or m not in p_p_m[id]:
                predict.append(0)
            else:
                predict.append(p_p_m[id][m])
            if id in movie_user_dic[m] and movie_user_dic[m][id] == 2:
                real.append(2)
            else:
                real.append(1)
    return predict, real

def ROCPlot(std, predict, path):
    std = np.array(std) 
    predict = np.array(predict)
    fpr, tpr, thresholds = metrics.roc_curve(std, predict, pos_label = 2)
    auc = metrics.auc(fpr, tpr)
    print "AUC= ", auc
    plt.plot(fpr, tpr, 'g-')
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.axis([0,1,0,1])
    plt.savefig(path)

def AsTask1GROC():
    test_path = 'data/roc/as/test.txt'
    movie_user_dic = LoadMovieUserDicForImplicitROC(test_path)
    p_p_m_path = 'data/roc/as/task1_p_p_m'
    p_p_m = LoadPPM(p_p_m_path)
    predict, real = ImplicitGROC(movie_user_dic, p_p_m)
    return real, predict

def AsTask2GROC():
    test_path = 'data/roc/as/test.txt'
    movie_user_dic = LoadMovieUserDicForImplicitROC(test_path)
    p_p_m_path = 'data/roc/as/task3_p_p_m'
    p_p_m = LoadPPM(p_p_m_path)
    predict, real = RatingPredictionGROC(movie_user_dic, p_p_m)
    return real, predict
 
def AsTask3GROC():
    test_path = 'data/roc/as/test.txt'
    rating = LoadTest(test_path)
    p_p_m_path = 'data/roc/as/task3_p_p_m'
    p_p_m = LoadPPM(p_p_m_path)
    predict, real = RatingImputationGROC(rating, p_p_m)
    return real, predict

def main():
    ImplicitGROCMain()
    RatingPredictionGROCMain()
    RatingImputationGROCMain()

#main()
