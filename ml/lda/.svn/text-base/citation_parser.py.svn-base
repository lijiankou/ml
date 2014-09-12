import numpy as np
import pickle

def ReadFileToString(path):
    file_object=open(path, 'r')
    content=file_object.read()
    file_object.close()
    return content

def ReadData(path):
    string = ReadFileToString(path)
    ll = string.split('\n\n')
    network = {}
    corpus = {}
    time = {}
    for str_p in ll:
        if len(str_p.strip()) == 0:
            continue
        str_ps = str_p.split('\n')
        p = {}
        citation = []
        title = ''
        abstract = ''
        id = ''
        t = ''
        for l in str_ps:
            if l.startswith("#*"):#title
                 title = l[(l.find("#*") + len('#*')):]
            if l.startswith('#%'):#citation
                 citation.append(int(l[(l.find("#%") + len('#%')):].strip()))
            if l.startswith('#index'):#index
                 id = int(l[(l.find("#index") + len('#index')):].strip())
            if l.startswith('#!'): #abstract
                 abstract = l[(l.find("#!") + len('#!')):]
            if l.startswith('#t'): #time
                 t = int(l[(l.find("#t") + len('#t')):])
        corpus[id] = abstract
        network[id] = citation
        time[id] = t
    return network, corpus, time

def DumpPickle(data, path):
    f = open(path, 'wb' )
    pickle.dump(data, f)

def LoadPickle(path):
    f = open(path, 'rb' )
    entry = pickle.load(f)
    return entry

#convert the original dic into new id, from 1 to len(dic)
#the original dic has been changed
def DicEncode(dic):
    num = 1
    new_dic = {}
    for i in dic:
        new_dic[i] = num
        num += 1
    return new_dic

#create corpus_dic from corpus
#return the docs with raw words and corpus_dic
def CorpusDic(doc, stopwords):
    dic = {}
    for k in doc:
        words = doc[k].split()
        result = {}
        for w in words:
            w = w.strip(' \t\n.\\/[]"\',$@%!{}&:-()?><0123456789').lower()
            if w.find('/') != -1:
                tmp = w.split('/')
                if tmp[1].isdigit:
                    continue
            if w.find('.') != -1:
                tmp = w.split('.')
                if tmp[1].isdigit() and tmp[0].isdigit():
                    continue
            if w.isdigit():
                continue
            if len(w) == 0:
                continue
            if w not in stopwords:
                result[w] = result.get(w, 0) + 1
        for w in result:
            dic[w] = dic.get(w, 0) + 1 
        doc[k] = result
    new_dic = {}
    for w in dic:
        if dic[w] > 10:
            new_dic[w] = dic[w]
    return new_dic, doc 
        
def LoadStopWords(path):
    o = open(path)
    dic = {}
    for l in o:
        dic[l.strip().lower()] = ''
    o.close()
    return dic
 
#convert the raw words to id
def CorpusEncode(cor, dic):
    print len(dic.keys())
    new_cor = {} 
    for k in cor:
        new_cor[k] = {}
        for l in cor[k]:
            if l in dic:
                new_cor[k][dic[l]] = cor[k][l]
    return new_cor

def SaveCorpus(corpus, path):
    o = open(path, 'wb')
    for doc in corpus:
        o.write(str(len(corpus[doc])))
        o.write(' ')
        for w in corpus[doc]:
            o.write(str(w))
            o.write(':')
            o.write(str(corpus[doc][w]))
            o.write(' ')
        o.write('\n')
    o.close()
 
def GetFinalNetworkAndCorpus(path):
    network, corpus, time = ReadData(path)
    #corpus_path = 'corpus_pickle'
    #DumpPickle(paper, paper_path)
    #network_path = 'network_pickle'
    #DumpPickle(doc, doc_path)

    stop_path = 'stopwords.txt'
    stop_dic = LoadStopWords(stop_path)
    #paper = LoadPickle(paper_path)
    #corpus = LoadPickle(doc_path)

    corpus_dic, corpus = CorpusDic(corpus, stop_dic)
    corpus_dic = DicEncode(corpus_dic)

    corpus = CorpusEncode(corpus, corpus_dic)
    return network, corpus, time

def SaveNetwork(network, path):
    o = open(path, 'wb')
    for i in network:
        for e in network[i]:
            o.write(str(i))
            o.write(' ')
            o.write(str(e))
            o.write('\n')
    o.close()

def FilterSmallData(corpus, network, sub_id, factor):
    r_cor = {}
    for i in corpus:
        if i in sub_id:
            r_cor[factor[i]] = corpus[i]
    r_network = {}
    for i in network:
        if i in sub_id:
            if i not in r_network:
                r_network[factor[i]] = []
            for j in network[i]:
                if j in sub_id:
                    r_network[factor[i]].append(factor[j])
    return r_network, r_cor 

def SelectId(corpus, time, t):
    r = {}
    for i in time:
        if time[i] < t and len(corpus[i]) != 0:
            r[i] = ''
    return r

def Factor(sub_id):
    i = 0
    r = {}
    for j in sub_id:
        r[j] = i
        i += 1
    return r

def main():
    path = 'outputacm.txt'
    path = '/home/lijk/working/project/data/bigdata/arnetminer/outputacm.txt'
    net, cor, time = GetFinalNetworkAndCorpus(path)
    sub_id = SelectId(cor, time, 1980)
    factor = Factor(sub_id)
    print len(sub_id)
    print len(factor)
    net, cor = FilterSmallData(cor, net, sub_id, factor)
    path = 'rtm_corpus'
    SaveCorpus(cor, path)
    path = 'rtm_network'
    SaveNetwork(net, path)
 
main()
