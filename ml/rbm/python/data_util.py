import random

def CreateTBMTrain(src, des):
    fs = open(src)
    fd = open(des, 'wb')
    for d in fs:
        t = d.split()
        if t[2] == '1':
            fd.write(' '.join(t))
            fd.write('\n')
    fs.close()
    fd.close()

def CreateTBMClickPredict(src, des):
    fs = open(src)
    fd = open(des, 'wb')
    for d in fs:
        t = d.split()
        t[2] ='1'
        fd.write(' '.join(t))
        fd.write('\n')
    fs.close()
    fd.close()

def DateTest(path):
    o = open(path)
    for l in o:
        s = l.split()
        if s[0] == '0' or s[1] == '0':
            print '0'
    o.close()
 
def LoadRating(path):
    o = open(path)
    dic = {}
    for l in o:
        t = l.split()
        user = int(t[0])
        item = int(t[1])
        if user not in dic:
            dic[user] = {}
        dic[user][item] = int(t[2])
    return dic
 
def SaveRating(rating, path):
    o = open(path, 'wb')
    for u in rating:
        for i in rating[u]:
            o.write('%d %d %d\n' %(u, i , rating[u][i]))
    o.close()

def CreateCRBMTask1Data(src, cold_dic, des, p):
    m_num = 1683
    rating = LoadRating(src)
    for u in rating:
        for i in range(1, m_num):
            if i in rating[u]:
                rating[u][i] = 1
            if i not in rating[u] and i not in cold_dic:
                if random.random() < p:
                    rating[u][i] = 0
    SaveRating(rating, des)
 
def CreateCRBMTask1Test(src, des):
    rating = LoadRating(src)
    for u in rating:
        for v in rating[u]:
            rating[u][v] = 1
    SaveRating(rating, des)
 
def CreateCRBMTask2Data(src, cold_dic, des, p):
    m_num = 1683
    rating = LoadRating(src)
    for u in rating:
        for i in range(1, m_num):
            if i in rating[u]:
                rating[u][i] = rating[u][i]
            if i not in rating[u] and i not in cold_dic:
                if random.random() < p:
                    rating[u][i] = 0
    SaveRating(rating, des)
  
def main():
    src = 'data/movielen_binary/train'
    des = 'data/movielen_binary/tbm_task3_train'
    CreateTBMTrain(src, des)
    src = 'data/movielen_binary/train'
    des = 'data/movielen_binary/tbm_task1_train'
    CreateTBMClickPredict(src, des)
    src = 'data/movielen_binary/test'
    des = 'data/movielen_binary/tbm_task1_test'
    CreateTBMClickPredict(src, des)
  
def LoadColdDic(path):
    o = open(path)
    dic = {}
    for l in o:
        t = l.split()
        dic[int(t[1])] = ''
    return dic

def CRBMMain():
    test_path = 'data/movielen_binary/test'
    cold_dic = LoadColdDic(test_path)
    des_path = 'data/movielen_binary/crbm_task1_test'
    CreateCRBMTask1Test(test_path, des_path)

    src = 'data/movielen_binary/train'
    des = 'data/movielen_binary/crbm_task1_train'
    p = 0.4
    CreateCRBMTask1Data(src, cold_dic, des, p)

    src = 'data/movielen_binary/train'
    des = 'data/movielen_binary/crbm_task2_train'
    CreateCRBMTask2Data(src, cold_dic, des, p)

#main()
CRBMMain()
