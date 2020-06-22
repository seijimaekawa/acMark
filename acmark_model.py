import numpy as np
from scipy import sparse
from scipy.stats import bernoulli
import random
import copy
import sys
import powerlaw
import warnings
warnings.simplefilter('ignore')

def node_deg(n,m,max_deg):
    p = 2
    simulated_data = [0]
    while sum(simulated_data)/2 < m:
        theoretical_distribution = powerlaw.Power_Law(xmin = 1., parameters = [p])
        simulated_data=theoretical_distribution.generate_random(n)
        for i in range(n):
            while simulated_data[i] >= max_deg:
                simulated_data[i] = theoretical_distribution.generate_random(1)[0]
            tmp = int(simulated_data[i]*2)
            if tmp % 2 != 0:
                tmp += 1
            simulated_data[i] = int(tmp/2)
        if (m - sum(simulated_data)/2) < m/10:
            p -= 0.01
        else:
            p -= 0.1
        if p<1.01:
            print("break")
            break
            
    theta = list(simulated_data)
    for i in range(n):
        theta[i] = float(theta[i])
    theta.sort(reverse=True)    
    return theta

def count_node_degree(S):
    n = S.shape[0]
    node_degree = np.zeros(n)
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            node_degree[nnz[0][i]] += 1
            node_degree[nnz[1][i]] += 1
    return int(sum(node_degree)/2)

def distribution_generator(flag, para_pow, para_normal, para_zip, t):
    if flag == "power_law":
        dist = 1 - np.random.power(para_pow, t) # R^{k}
    elif flag == "uniform":
        dist = np.random.uniform(0,1,t)
    elif flag == "normal":
        dist = np.random.normal(0.5,para_normal,t)
    elif flag == "zipfian":
        dist = np.random.zipf(para_zip,t)
    return dist


def class_generation(n, k, alpha, phi_c):
    chi = distribution_generator("power_law",phi_c,0,0, k)
#     chi = distribution_generator("normal",phi_c,0,0, k)
    
    chi = np.array(chi) * alpha / sum(chi)
    U = np.random.dirichlet(chi, n)
    C = [] # class assignment list (finally, R^{n})
    for i in range(n):
        C.append(np.argmax(U[i]))

    counter=[];x=[]
    for i in range(k):
        x.append(i)
        counter.append(C.count(i))
    print("class size disribution : ",end="")
    print(counter)
    if 0 in counter:
        print('Error! There is a class which has no member.')
        sys.exit(1)

    return U,C

def adjust(n,k,U,C,density):
    U_prime = copy.deepcopy(U)
    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(C)):
        partition[C[i]].append(i)
        
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    def inverse(U_tmp):
        U_ = 1 - U_tmp
        U_ /= sum(U_tmp)
        return U_
    
    for l in range(k):
        Th=1
        if  density[l] >= 1/k:
            while True:
                Th -= 0.1
                max_entry_value = 0
                for j in partition[l]:
                    max_entry_value += freez_func(U[j],Th)[l]**2
                if density[l] < max_entry_value/len(partition[l]):
                    break
                if Th <= 0:
                    print("break1")
                    Th=0.01
                    break
            for j in partition[l]:
                U[j] = freez_func(U[j],Th)
                U_prime[j] = U[j]
        else:
            while True:
                Th -= 0.1
                max_entry_value = 0
                for j in partition[l]:
                    U_tmp = freez_func(U[j],Th)
                    max_entry_value += U_tmp[l]*inverse(U_tmp)[l]
                if density[l] > max_entry_value/len(partition[l]):
                    break
                if Th <= 0:
                    print("break2")
                    Th=0.01
                    break
            for j in partition[l]:
                U[j] = freez_func(U[j],Th)
                U_prime[i] = inverse(U[j])
#         print(Th)  
    return U, U_prime
        
def edge_construction(n, U, k, U_prime, step, theta, r):
    U_ = copy.deepcopy(U)
    
    S = sparse.dok_matrix((n,n))
    degree_list = np.zeros(n)
    count_list = []

    print_count = 1
    for i in range(n):
        if i/n * 10 > print_count:
            print("finished " +str(print_count)+"0%")
            print_count += 1
        count = 0
        ng_list = set([i])
        while count < r and degree_list[i] < theta[i]:
            to_classes = random.choices(list(range(0,k)), k=int(theta[i]-degree_list[i]), weights=U_[i])
            for to_class in to_classes:
                for loop in range(50):
                    j = U_prime[to_class][int(random.random()/step)]
                    if j not in ng_list:
                        ng_list.add(j)
                        break
                if degree_list[j] < theta[j] and i!=j:
                    S[i,j] = 1;S[j,i] = 1
                    degree_list[i]+=1;degree_list[j]+=1
            count += 1 
        count_list.append(count)
    return S, count_list

def ITS_U_prime(n,k,U_prime):
    class_list = []
    step = 1/(n*100)
    UT = U_prime.transpose()
    for i in range(k): # クラスタごとに分布を作成
        UT_tmp = UT[i]/ sum(UT[i])
        for j in range(n-1):
            UT_tmp[j+1] += UT_tmp[j]

        class_tmp = []
        node_counter = 0
        for l in np.arange(0,1,step):
            if UT_tmp[node_counter] > l:
                class_tmp.append(node_counter)
            else:
                node_counter += 1
                class_tmp.append(node_counter)
        class_list.append(class_tmp)
    return class_list

def attribute_generation(n,d,k,U,C,beta,sigma,omega):
    chi = distribution_generator("normal", 0, sigma, 0, k)
    V = np.random.dirichlet(chi*beta, d).T
    X = U@V

    def variation_attribute(n,d,k,X,C,att_flag="normal"):
        if att_flag == "normal":
            for i in range(d): # each attribute demension
                clus_dev = np.random.uniform(omega,omega,k) # variation for each class
                for p in range(n): # each node
                    X[p,i] += np.random.normal(0.0,clus_dev[C[p]],1)    
        else: # Bernoulli distribution
            for i in range(d):
                for p in range(n):
                    X[p,i] = bernoulli.rvs(p=X[p,i], size=1)        
        # normalization
        for i in range(d):
            X[:,i] -= min(X[:,i])
            X[:,i] /= max(X[:,i])
 
        return X
    
    return variation_attribute(n,d,k,X,C)
    
    
def acmark(n,m,k,d,max_deg,density,alpha=1,phi_c=1,beta=0.1,sigma=0,omega=0.2,r=50):
    # node degree generation 
    theta = node_deg(n,m,max_deg)

    # class generation
    U,C = class_generation(n,k,alpha,phi_c)
    
    # adjusting phase
    U,U_prime = adjust(n,k,U,C,density)
    
    # Inverse Transform Sampling
    U_prime_CDF = ITS_U_prime(n,k,U_prime)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, 1/(n*100), theta, r)
    print("number of generated edges : " + str(count_node_degree(S_gen)))
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,C,beta,sigma,omega)
    
    return S_gen,X,C

######### for reproduction #########

def class_reproduction(n,k,S,Label):
    # extract class preference matrix from given graph
    pref = np.zeros((len(Label),k))
    nnz = S.nonzero()
    for i in range(len(nnz[0])):
        if nnz[0][i] < nnz[1][i]:
            pref[nnz[0][i]][Label[nnz[1][i]]] += 1
            pref[nnz[1][i]][Label[nnz[0][i]]] += 1
    for i in range(len(Label)):
        pref[i] /= sum(pref[i])

    partition = []
    for i in range(k):
        partition.append([])
    for i in range(len(Label)):
        partition[Label[i]].append(i)
        
    # caluculate average and deviation of class preference
    from statistics import mean, median,variance,stdev
    class_pref_ave = np.zeros((k,k))
    class_pref_stdev = np.zeros((k,k))
    for i in range(k):
        pref_tmp = []
        for j in partition[i]:
            pref_tmp.append(pref[j])
        pref_tmp = np.array(pref_tmp).transpose()
        for h in range(k):
            class_pref_ave[i,h] = mean(pref_tmp[h])
            class_pref_stdev[i,h] = stdev(pref_tmp[h])
            
    density = np.zeros(k)
    for l in range(k):
        density[l] = class_pref_ave[l,l]
    
    # generate U from class preference matrix
    # Freezing function
    def freez_func(q,Th):
        return q**(1/Th) / np.sum(q**(1/Th))
    
    com_size = []
    for i in partition:
        com_size.append(len(i))
    com_size = np.array(com_size) / sum(com_size)
    
    U = np.zeros((n,k))
    C=[]
    for i in range(n):
        C_tmp = random.choices(list(range(0,k)),k=1,weights=com_size)[0]
        C.append(C_tmp)
        for h in range(k):
            U[i,h] = np.random.normal(loc=class_pref_ave[C_tmp][h],scale=class_pref_stdev[C_tmp][h],size=1)[0]

    # eliminate U<0 and U>1 (keep 0<=U<=1)
    minus_list = np.where(U < 0)
    for i in range(len(minus_list[0])):
        U[minus_list[0][i],minus_list[1][i]] = 0
    one_list = np.where(U > 1)
    for i in range(len(one_list[0])):
        U[one_list[0][i],one_list[1][i]] = 1
    # normalize
    for i in range(n):
        U[i] /= sum(U[i])

    return U,C,density


def acmark_reproduction(S,Label,d,n=0,m=0,max_deg=0,beta=0.1,sigma=0,omega=0.2,r=50):

    # node degree generation 
    if n == 0:
        theta = np.zeros(len(Label))
        nnz = S.nonzero()
        for i in range(len(nnz[0])):
            if nnz[0][i] < nnz[1][i]:
                theta[nnz[0][i]] += 1
                theta[nnz[1][i]] += 1
    else:
        theta = node_deg(n,m,max_deg)
    n = len(theta)
    m = count_node_degree(S)
    k = len(set(Label))
    
    # class generation
#     U,C = class_generation(n,k,alpha,phi_c)
    U,C,density = class_reproduction(n,k,S,Label)
    
    # adjusting phase
    U,U_prime = adjust(n,k,U,C,density)
    
    # Inverse Transform Sampling
    U_prime_CDF = ITS_U_prime(n,k,U_prime)

    # Edge generation
    S_gen, count_list = edge_construction(n, U, k, U_prime_CDF, 1/(n*100), theta, r)
    print("number of generated edges : " + str(count_node_degree(S_gen)))
    
    # Attribute generation
    X = attribute_generation(n,d,k,U,C,beta,sigma,omega)
    
    return S_gen,X,C