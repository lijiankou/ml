import matplotlib.pyplot as plt
import os

def CreateList(num):
    res = []
    for i in range(num):
        res.append(0)
    return res

def LoadData(path):
    fr = open(path)
    data = []
    for l in fr:
        if l.strip() > 0: 
            t = l.split()
            num = []
            for n in t:
                num.append(int(n))
            if len(num) > 0:
                data.append(num)
    return data

def Calculate(data):
    t_p = 0
    t_n = 0
    for d in data:
        for t in d:
            if t == 2: 
                t_p += 1
            else:
                t_n += 1
    return t_p, t_n

def MaxColSize(data):
    max_len = 0.0
    for d in data:
        if len(d) > max_len:
            max_len = len(d) 
    return max_len

def CROC(real):
    t_p, t_n = Calculate(real)
    tr = []
    fr = []
    max_len = MaxColSize(real)
    sum_t = 0
    sum_f = 0
    for i in range(max_len):
        for j in range(len(real)):
            if len(real[j]) > i:
                if real[j][i] == 2:
                    sum_t += 1
                else:
                    sum_f += 1
        tr.append(float(sum_t)/t_p)
        fr.append(float(sum_f)/t_n)
    return tr, fr

def CalculateTotalPosAndNeg(data):
    t_p = []
    t_n = []
    for i in range(len(data)):
        t_p_i = 0
        t_n_i = 0
        for t in data[i]:
            if t == 2: 
                t_p_i += 1
            else:
                t_n_i += 1
        t_p.append(t_p_i)
        t_n.append(t_n_i)
    return t_p, t_n

def CROCMovie(num, real):
    t_p, t_f = CalculateTotalPosAndNeg(real)
    for j in range(num):
        tr = []
        fr = []
        sum_t = 0
        sum_f = 0
        print real[num]
        for i in range(len(real[j])):
            if len(real[j]) > i:
                if real[j][i] == 2:
                    sum_t += 1
                else:
                    sum_f += 1
            tr.append(float(sum_t) / t_p[j])
            fr.append(float(sum_f) / t_f[j])
        PlotCROC(fr, tr, 'croc.eps')

def CROCAvg(real):
    t_p, t_f = CalculateTotalPosAndNeg(real)
    tr = []
    fr = []
    max_len = MaxColSize(real)
    sum_t = CreateList(len(real))
    sum_f = CreateList(len(real))

    count = 0
    for j in range(len(real)): 
        if t_p[j] == 0 or t_f[j] == 0:
            continue
        count += 1

    for i in range(max_len):
        for j in range(len(real)):
            if len(real[j]) > i:
                if real[j][i] == 2:
                    sum_t[j] += 1
                else:
                    sum_f[j] += 1
        sum_tt = 0
        sum_ff = 0
        for j in range(len(sum_t)): 
            if t_p[j] == 0 or t_f[j] == 0:
                continue
            sum_tt += float(sum_t[j]) / t_p[j]
            sum_ff += float(sum_f[j]) / t_f[j]
        tr.append(float(sum_tt)/count)
        fr.append(float(sum_ff)/count)
    return tr, fr

#pr for x, tr for y 
def PlotCROC(pr, tr, path):
    plt.plot(pr, tr,'g-')
    plt.ylabel("True Positive Rate")
    plt.title("IROC Curve")
    plt.xlabel("False Positive Rate")
    plt.axis([0,1,0,1])
    plt.savefig(path)
  
def main():
    path = 'data/roc/crbm_croc'
    path = 'data/roc/tbm_task3_iroc'
    path = 'data/roc/tbm_task2_iroc'
    path = 'data/roc/tbm_task1_iroc'
    data = LoadData(path)
    #tr, fr = CROC(data)
    tr, fr = CROCAvg(data)
    fig_dir = 'data/figure/'
    name = 'tbm_task3_iroc.eps'
    name = 'tbm_task2_iroc.eps'
    name = 'tbm_task1_iroc.eps'
    PlotCROC(fr, tr, os.path.join(fig_dir, name))
    #CROCMovie(len(data), data)
  
def CRBMmain():
    path = 'data/roc/crbm/crbm_task1_iroc'
    path = 'data/roc/crbm/crbm_task3_iroc'
    path = 'data/roc/crbm/crbm_task2_iroc'
    data = LoadData(path)
    tr, fr = CROCAvg(data)
    fig_dir = 'data/figure/crbm/'
    name = 'crbm_task2_iroc.eps'
    PlotCROC(fr, tr, os.path.join(fig_dir, name))
    name = 'crbm_task1_iroc.eps'
    PlotCROC(fr, tr, os.path.join(fig_dir, name))

#main()
CRBMmain()
