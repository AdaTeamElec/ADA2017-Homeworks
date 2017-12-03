from sklearn.cluster import KMeans
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

### -------------  Signed Graph -------------

def get_w_signed(W_coop, W_comp, id_groups, binary=False):
    W = W_coop - W_comp
    W = W.toarray()[id_groups, :]
    W = W[:, id_groups]
    if binary:
        W[W > 0] = 1
        W[W < 0] = -1
    return csr_matrix(W)

def compute_L(W, normalized=False):
    ''' 
    Compute Laplacian of signed graph given by W. Return L 
    '''
    if normalized:
        N = W.shape[0]
        d = np.power(np.sum(np.abs(W), axis=1), -0.5)
        D = scipy.sparse.diags(np.ravel(d), 0).tocsc()
        L = scipy.sparse.identity(N) - D * W * D
    else:
        D = scipy.sparse.diags(np.ravel(np.abs(W).sum(1)), 0)
        L = (D - W).tocsc()
    return L

def get_basis_L(L, k=3):
    ''' 
    Return approximation of Laplacian k first eigenvalues and eigenvectors 
    '''
    lam, V = scipy.sparse.linalg.eigsh(L, k=k, which='SM')
    return lam, V

def get_clusters(L, k=3):
    '''
    Cluster nodes using eigenvector of Laplacian as feature with kMean. Return
    labels of cluster.
    '''    
    lam, V = get_basis_L(L, k)
    if len(np.nonzero(lam > 1e-10)[0]) == 0:
        X = V[:, [0]]
    else :
        id_first = np.nonzero(lam > 1e-10)[0][0]
        X = V[:,id_first:]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    return kmeans.labels_

def cut_loss(W, cgt):
    '''
    Compute loss function for clustered graph. For each cluster sum internal
    loss (negative link inside cluster) and external loss (positive link to 
    other cluster). W is the adjacency and cgt the ground truth
    '''
    if cgt is None:
        return np.nan
    nk = np.unique(cgt)
    Ap = np.maximum(W.toarray(), 0)
    An = np.maximum(-W.toarray(), 0)
    cut = 0
    for k in nk:
        id_x = cgt == k
        id_y = cgt != k
        # Count positive links with other cluster
        cutp = np.sum(Ap[id_x, :][:, id_y])
        # Count negative links inside< cluster
        cutn = np.sum(An[id_x, :][:, id_x])
        lambda_ = (1/np.sum(id_x) + 1/np.sum(id_y))*len(id_x)
        cut += lambda_*(cutp + cutn)
    cut = cut/len(nk)
    return cut

def estimate_ncluster(W, L, plotloss=False):
    '''
    Try to cluster with multiple k, and return the one with smallest loss
    '''
    n_max = L.shape[0]-1
    loss = np.zeros((n_max, 2))

    for i, k in enumerate(range(1,n_max+1)):
        cgt = get_clusters(L, k)
        loss[i, 0] = k
        loss[i, 1] = cut_loss(W, cgt)
    
    if plotloss:
        plt.plot(loss[:,0], loss[:, 1])
    
    return int(loss[np.argmin(loss[:,1]), 0])

def reindex_W_with_classes(W,C):
    """
    Function that reindexes W according to communities/classes

    Usage:
      [reindexed_W,reindexed_C] = reindex_W_with_C(W,C)

    Notations:
      n = nb_data
      nc = nb_communities

    Input variables:
      W = Adjacency matrix. Size = n x n.
      C = Classes used for reindexing W. Size = n x 1. Values in [0,1,...,nc-1].

    Output variables:
      reindexed_W = reindexed adjacency matrix. Size = n x n.
      reindexed_C = reindexed classes C. Size = n x 1. Values in [0,1,...,nc-1].
    """

    n = C.shape[0] # nb of vertices
    nc = len(np.unique(C)) # nb of communities
    reindexing_mapping = np.zeros([n]) # mapping for reindexing W
    reindexed_C = np.zeros([n]) # reindexed C
    tot = 0
    for k in range(nc):
        cluster = (np.where(C==k))[0]
        length_cluster = len(cluster)
        x = np.array(range(tot,tot+length_cluster))
        reindexing_mapping[cluster] = x
        reindexed_C[x] = k
        tot += length_cluster
        
    idx_row,idx_col,val = scipy.sparse.find(W)
    idx_row = reindexing_mapping[idx_row]
    idx_col = reindexing_mapping[idx_col]
    reindexed_W = scipy.sparse.csr_matrix((val, (idx_row, idx_col)), shape=(n, n))

    return reindexed_W,reindexed_C


### -------------  DRAWING -------------

def get_dummy(type_=0):
    '''
    Get dummy graph for debug or proof of concept. type=1 is graph of
    positively connected exept one edge. type=0 
    is splited graph (yes -> yes -> no -> yes). Return adjacency matrix
    '''
    
    if type_ == 0:
        W = np.zeros((14,14))
        np.fill_diagonal(W[1:, 0:], 1, wrap=True)
        W[-1, 0] = 1; W[3, 2] = -1; W[7, 6] = -1; W[-1, -2] = -1; 
        W = W + W.T
    else:
        W = np.zeros((7,7))
        np.fill_diagonal(W[1:, 0:], 1, wrap=True)
        np.fill_diagonal(W[0:, 1:], 1, wrap=True)
        W[0, -1] = -1
        W[-1, 0] = -1
    W = scipy.sparse.csr_matrix(W) 
    return W

def get_pos(W, cgt, r=3):
    '''
    Take adjacency matrix (W) and ground truth for nodes (cgt). Will create
    a graph with prdefined structure (star graph). The radius can be changed with 
    r parameter. Return position 2d vector
    '''
    
    # Get id of unique cluster
    n_cluster = len(np.unique(cgt))
    # Split full angle circle (2*pi) with cluster number
    angle = (2*np.pi/n_cluster)*np.arange(n_cluster)
    # Set base point for cluster location
    center = np.array([ r*np.cos(angle), r*np.sin(angle)]).T
    pos = np.zeros((len(cgt), 2))
    # Generate cluster position
    for i in range(n_cluster):
        n_sub = np.sum(cgt == i)
        angle = (2*np.pi/n_sub)*np.arange(n_sub)
        pos[cgt == i] = center[i] + np.array([np.cos(angle), np.sin(angle)]).T
    
    return pos


def draw_graph(W, cgt=None, reorder=False, labels=None, ax=None, offset=0):
    '''
    Draw graph for viz. If cgt is not present nodes will be set to black. If cgt 
    is given node will be cluster acordingly. Labels can be used for better display
    Param:
        W (sparse): is the adjacency matriy of the graph
        cgt (array 1d): ground truch for nodes
        labels (array 1d): string array of label to display above name
    ''' 

    # Build graph fro W
    G = nx.from_scipy_sparse_matrix(W)
    val = [-W[g[0],g[1]] for g in G.edges]
    # Compute Laplacian to get spatial representation
    L = compute_L(W, normalized=True)
    lam, V = get_basis_L(L)
    id_first = np.nonzero(lam > 1e-10)[0][0]
    base_g = V[:,id_first:id_first+2]
    if reorder and cgt is not None:
        base_g = get_pos(W, cgt)
    
    # Plot graph
    if ax is None:
        plt.figure(figsize=(12,12))
        ax_ = plt.subplot(1,1,1)
    else:
        ax_ = ax

    if cgt is not None:
        nx.draw_networkx_nodes(G, pos=base_g, node_color=cgt, node_cmap=plt.cm.hsv, ax=ax_)
    else:
        nx.draw_networkx_nodes(G, pos=base_g, node_color='k', ax=ax_)

    nx.draw_networkx_edges(G, pos=base_g, edge_color=val, width=4, 
                           edge_cmap=plt.cm.bwr, edge_vmax=1, edge_vmin=-1, ax=ax_)
    
    # nx.draw_networkx_labels(G, base_g, font_color='w', ax=ax_)
    if labels is not None:
        d = dict(zip(np.arange(len(labels)), labels))
        base_g_lab = base_g
        base_g_lab[:, 1] += offset
        nx.draw_networkx_labels(G, base_g, d, font_size=9, ax=ax_)
    ax_.axis('off')
    
        