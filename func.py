import numpy as np

def calc_class_features(S,k,Label):
    pref = np.zeros((len(Label),k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])
    pref = np.nan_to_num(pref)

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    class_pref_mean = np.zeros((k,k))
    class_pref_dev = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_mean[i,h] = mean(pref_tmp[h])
            class_pref_dev[i,h] = stdev(pref_tmp[h])

    return class_pref_mean, class_pref_dev


def S_class_order(S, n, k, Label):
    import scipy.sparse as sp
    import random
    import copy
    partition = []
    k = max(Label)+1
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)

    for i in range(k):
        random.shuffle(partition[i])

    community_size = []
    for i in range(len(partition)):
        community_size.append(len(list(partition)[i]))
    print ("community size : " + str(community_size))
    com_size_dict = {}
    for com_num, size in enumerate(community_size):
        com_size_dict[com_num] = size
    com_size_dict = dict(sorted(com_size_dict.items(), key=lambda x:x[1],  reverse=True))
    print(com_size_dict)

    communities = copy.deepcopy(partition)
    partition = []
    for com_num in com_size_dict.keys():
        for node in list(communities)[com_num]:
               partition.append(node)
    print(len(partition))

    import random
    S_class = sp.dok_matrix((n,n))

    part_dic = {}
    for i in range(n):
        part_dic[partition[i]] = i

    nzs = S.nonzero()
    for i in range(len(nzs[0])):
        S_class[part_dic[nzs[0][i]],part_dic[nzs[1][i]]] = 1

    return S_class