# coding: utf-8
# # Core Program

import numpy as np
from scipy import sparse
import random
import scipy.io as sio
from scipy.stats import bernoulli

# ## You can see the details of the inputs on README

# ## Outputs
# ##### 1: adjacency matrix
# ##### 2: attribute matrix
# ##### 3: cluster assignment vector


def derived_from_dirichlet(n, m, d, k, k2, alpha, beta, gamma, node_d, com_s, phi_d, phi_c, sigma_d, sigma_c, delta_d, delta_c, att_power, att_uniform, att_normal):
    def selectivity(pri=0, node_d=0, com_s=0):
        # priority
        priority_list = ["edge", "degree"]
        priority = priority_list[pri]
        # node degree
        node_degree_list = ["power_law", "uniform", "normal", "zipfian"]
        node_degree = node_degree_list[node_d]
        # community size
        com_size_list = ["power_law", "uniform", "normal", "zipfian"]
        com_size = com_size_list[com_s]
        return priority, node_degree, com_size

    # (("edge","degree"),("power_law","uniform","normal","zipfian"),("power_law","uniform","normal","zipfian"))
    priority, node_degree, com_size = selectivity(node_d=node_d, com_s=com_s)

    # ## distribution generator

    def distribution_generator(flag, para_pow, para_normal, para_zip, t):
        if flag == "power_law":
            dist = 1 - np.random.power(para_pow, t)  # R^{k}
        elif flag == "uniform":
            dist = np.random.uniform(0, 1, t)
        elif flag == "normal":
            dist = np.random.normal(0.5, para_normal, t)
        elif flag == "zipfian":
            dist = np.random.zipf(para_zip, t)
        return dist

    # ## generate a community size list

    def community_generation(n, k, com_size, phi_c, sigma_c, delta_c):
        chi = distribution_generator(com_size, phi_c, sigma_c, delta_c, k)
        # chi = chi*(chi_max-chi_min)+chi_min
        chi = chi * alpha / sum(chi)
    # Cluster assignment
        # print(chi)
        U = np.random.dirichlet(chi, n)  # topology cluster matrix R^{n*k}
        return U

    # ## Cluster assignment

    U = community_generation(n, k, com_size, phi_c, sigma_c, delta_c)

    # ## Edge construction

    # ## node degree generation

    def node_degree_generation(n, m, priority, node_degree, phi_d, sigma_d, delta_d):
        theta = distribution_generator(node_degree, phi_d, sigma_d, delta_d, n)
        if priority == "edge":
            theta = np.array(list(map(int, theta * m * 2 / sum(theta) + 1)))
        # else:
        #     theta = np.array(list(map(int,theta*(theta_max-theta_min)+theta_min)))
        return theta

    theta = node_degree_generation(
        n, m, priority, node_degree, phi_d, sigma_d, delta_d)

    # Attribute generation

    num_power = int(d*att_power)
    num_uniform = int(d*att_uniform)
    num_normal = int(d*att_normal)
    num_random = d-num_power-num_uniform-num_normal

    # Input4 for attribute

    beta_dist = "normal"  # 0:power-law, 1:normal, 2:uniform
    gamma_dist = "normal"  # 0:power-law, 1:normal, 2:uniform
    phi_V = 2
    sigma_V = 0.1
    delta_V = 0.2
    phi_H = 2
    sigma_H = 0.1
    delta_H = 0.2

    # generating V
    chi = distribution_generator(beta_dist, phi_V, sigma_V, delta_V, k2)
    chi = np.array(chi)/sum(chi)*beta
    # attribute cluster matrix R^{d*k2}
    V = np.random.dirichlet(chi, num_power+num_uniform+num_normal)
    # generating H
    chi = distribution_generator(gamma_dist, phi_H, sigma_H, delta_H, k2)
    chi = np.array(chi)/sum(chi)*gamma
    H = np.random.dirichlet(chi, k)  # cluster transfer matrix R^{k*k2}

    return U, H, V, theta


def acmark(outpath="", n=1000, m=4000, d=100, k=5, k2=10, r=10, alpha=0.2, beta=10, gamma=1., node_d=0, com_s=0, phi_d=3, phi_c=2, sigma_d=0.1, sigma_c=0.1, delta_d=3, delta_c=2, att_power=0.0, att_uniform=0.0, att_normal=0.5, att_ber=0.0, dev_normal_max=0.3, dev_normal_min=0.1, dev_power_max=3, dev_power_min=2, uni_att=0.2):

    U, H, V, theta = derived_from_dirichlet(n, m, d, k, k2, alpha, beta, gamma, node_d, com_s,
                                            phi_d, phi_c, sigma_d, sigma_c, delta_d, delta_c, att_power, att_uniform, att_normal)
    C = []  # cluster list (finally, R^{n})
    for i in range(n):
        C.append(np.argmax(U[i]))
    r *= k

    def edge_construction(n, U, theta, around=1.0, r=10*k):
        # list of edge construction candidates
        S = sparse.dok_matrix((n, n))
        degree_list = np.zeros(n)
        # print_count=0
        for i in range(n):
            # if 100*i/n+1 >= print_count:
            #     print(str(print_count)+"%",end="\r",flush=True)
            #     print_count+=1
            count = 0
            while count < r and degree_list[i] < theta[i]:
                # step1 create candidate list
                candidate = []
                n_candidate = theta[i]
                candidate = np.random.randint(
                    0, n-1, size=n_candidate)  # for undirected graph
                candidate = list(set(candidate))

                # step2 create edges
                for j in candidate:
                    if i < j:
                        i_ = i
                        j_ = j
                    else:
                        i_ = j
                        j_ = i
                    if i_ != j_ and S[i_, j_] == 0 and degree_list[i_] < around * theta[i_] and degree_list[j_] < around * theta[j_]:
                        S[i_, j_] = np.random.poisson(U[i_, :].transpose().dot(
                            U[j_, :]), 1)  # ingoring node degree
                        if S[i_, j_] > 0:
                            degree_list[i_] += 1
                            degree_list[j_] += 1
                count += 1
        return S

    S = edge_construction(n, U, theta)

    ### Attribute Generation ###

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    ### Construct attribute matrix X from latent factors ###
    X = U.dot(H.dot(V.T))
    # X = sigmoid(U.dot(H)).dot(W.transpose())

    # ### Variation of attribute
    num_bernoulli = int(d*att_ber)
    num_power = int(d*att_power)
    num_uniform = int(d*att_uniform)
    num_normal = int(d*att_normal)
    num_random = d-num_power-num_uniform-num_normal

    def variation_attribute(n, k, X, C, num_bernoulli, num_power, num_uniform, num_normal, dev_normal_min, dev_normal_max, dev_power_min, dev_power_max, uni_att):
        for i in range(num_bernoulli):
            for p in range(n):
                X[p, i] = bernoulli.rvs(p=X[p, i], size=1)
        dim = num_bernoulli

        for i in range(num_power):  # each demension
            clus_dev = np.random.uniform(dev_normal_min, dev_normal_max-0.1, k)
            exp = np.random.uniform(dev_power_min, dev_power_max, 1)
            for p in range(n):  # each node
                # clus_dev[C[p]]
                X[p, i] = (X[p, i] * np.random.normal(1.0,
                                                      clus_dev[C[p]], 1)) ** exp

        dim += num_power
        for i in range(dim, dim+num_uniform):  # each demension
            clus_dev = np.random.uniform(1.0-uni_att, 1.0+uni_att, n)
            for p in range(n):
                X[p, i] *= clus_dev[C[p]]

        dim += num_uniform
        for i in range(dim, dim+num_normal):  # each demension
            clus_dev = np.random.uniform(dev_normal_min, dev_normal_max, k)
            for p in range(n):  # each node
                X[p, i] *= np.random.normal(1.0, clus_dev[C[p]], 1)
        return X

    ### Apply probabilistic distributions to X ###

    X = variation_attribute(n, k, X, C, num_bernoulli, num_power, num_uniform, num_normal,
                            dev_normal_min, dev_normal_max, dev_power_min, dev_power_max, uni_att)

    # random attribute
    def concat_random_attribute(n, X, num_random):
        rand_att = []
        for i in range(num_random):
            random_flag = random.randint(0, 2)
            if random_flag == 0:
                rand_att.append(np.random.normal(0.5, 0.2, n))
            elif random_flag == 1:
                rand_att.append(np.random.uniform(0, 1, n))
            else:
                rand_att.append(1.0-np.random.power(2, n))
        return np.concatenate((X, np.array(rand_att).T), axis=1)

    if num_random != 0:
        X = concat_random_attribute(n, X, num_random)

    # ## Regularization for attributes

    for i in range(d):
        X[:, i] -= min(X[:, i])
        X[:, i] /= max(X[:, i])

    if outpath != "":
        sio.savemat(outpath, {'S': S, 'X': X, 'C': C})
    else:
        return S, X, C

# acmark(outpath="test.mat")
