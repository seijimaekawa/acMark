import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

def load_data(file_name):
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)['arr_0'].item()
        S = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                              loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            _X_obs = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                   loader['attr_indptr']), shape=loader['attr_shape'])
        else:
            _X_obs = None

        Label = loader.get('labels')


    S= S + S.T
    S[S > 1] = 1
    lcc = largest_connected_components(S)
    S = S[lcc,:][:,lcc]
    Label = Label[lcc]
    n = S.shape[0]
    k = len(set(Label))

    for i in range(n):
        S[i,i] = 0
    nonzeros = S.nonzero()
    m = int(len(nonzeros[0])/2)
    print ("number of nodes : " + str(n))
    print ("number of edges : " + str(m))
    print ("number of classes : " + str(len(set(Label))))
    
    return S,Label,n,m,k


def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

