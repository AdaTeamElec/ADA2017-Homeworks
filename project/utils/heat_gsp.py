import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.collections as coll


def locate_node(row, gname, nodes, cls_groups):
    # Check if name makes sense and if group present in cls_groups
    if pd.isnull(row[gname]) or row[gname] not in cls_groups:
        return (-1, -1), -1
    # Get groups position in index and node concerned
    node_id =  nodes.loc[row.id_loc, 'index']
    group_id = np.where(cls_groups == row[gname])[0][0]
    return (node_id, group_id), row.fillna(0).nkill


def get_signal_attack(df_time, nodes, cls_groups):
    # Define two separeate signals. One for frequency of attacks at this poin, the other for 
    # the number of casualities
    signal_frequ = np.zeros((len(nodes), len(cls_groups)))
    signal_kill = np.zeros((len(nodes), len(cls_groups)))
    tags = ['gname', 'gname2', 'gname3']
    # Iterate over all the events
    for event in df_time.index:
        # Iterate over the 3 groups
        for tag in tags:
            # Get node/groups coordinate and number of kills
            (x,y), z = locate_node(df_time.loc[event], tag, nodes, cls_groups)
            if any(np.array([x,y,z]) == -1):
                continue
            # Add frequency for node and number of kill
            signal_frequ[x, y] += 1
            signal_kill[x, y] += z
    # Use loigarithm representation to avoid huge number (stay fair)
    return np.log(1+signal_frequ) + np.log(1+signal_kill)


def knn_graph(cloud, k=20, mode='local'):

    coord = cloud[['latitude', 'longitude']].values
    # Compute kNN and fit data
    nn = NearestNeighbors(n_neighbors=k)
        
    nn.fit(coord) 
    dists, ids = nn.kneighbors(coord)
        
    #print(np.mean(dists[:,1:]), np.median(dists[:,1:]), np.max(dists[:,1:]), np.min(dists[:,1:]))
    s = np.median(dists[:,1:])
    w = np.exp(-np.power(dists,2)/(2*np.power(s,2)))

    rows, _ = np.indices(np.shape(ids))
    if mode == 'connectivity':
        w = np.ones(np.shape(dists))
    elif mode == 'distance':
        #print(np.mean(dists[:,1:]), np.median(dists[:,1:]), np.max(dists[:,1:]), np.min(dists[:,1:]))
        s = np.median(dists[:,1:])
        w = np.exp(-np.power(dists,2)/(2*np.power(s,2)))
    elif mode == 'local':
        # Check construction mode
        dists_ki = np.zeros(dists.shape) 
        dists_kj = np.zeros(dists.shape) 

        dists_ki = np.repeat([dists[:, -1]], dists.shape[1], axis=0).T + 1e-10
        dists_kj = dists[:, -1][ids] + 1e-10
        w = np.exp(-np.power(dists,2)/(dists_ki*dists_kj))
    else:
        return
    
    # Do not take base point (same point in knn)
    keep = np.array([id_!= i for i, id_ in enumerate(ids)])
    # Complete matrix according to positions
    _W = coo_matrix((w[keep].flatten(), (ids[keep].flatten(), rows[keep].flatten())), 
                       shape=(np.shape(dists)[0], np.shape(dists)[0]))

    _W = 1/2*(_W + _W.T)
    
    return _W


def plot_map_group(nodes, signal, title='', bbox=[-18, 10, 65, 45]):

    plt.figure(figsize=(16,16))
    bbox = [-18, 10, 65, 45]
    map_ = Basemap(llcrnrlon=bbox[0], llcrnrlat=bbox[1], urcrnrlon=bbox[2], urcrnrlat=bbox[3], 
                   resolution='i', lat_0 = 0, lon_0 = 0)

    map_.drawmapboundary(fill_color='#d4dadc')
    map_.fillcontinents(color='#fafaf8', lake_color='#d4dadc')
    map_.drawcountries()

    x, y = map_(nodes.longitude.values, nodes.latitude.values)
    s_centerd = (signal-np.min(signal))/(np.max(signal)-np.min(signal))
    cs = map_.scatter(x, y, c=signal, cmap='nipy_spectral', zorder=3, s=5)
    cbar = map_.colorbar(cs,location='bottom',pad="5%")
    cbar.set_label('Estimation of attacks intensity')
    plt.title(title)
    plt.show()
    

def plot_map_cls(nodes, signal, cls_groups, title='', bbox=[5, 10, 65, 45]):

    id_keep = np.nonzero(signal)[0]
    signal = signal[id_keep]
    u = np.unique(signal)
    signal = [np.where(u==item)[0][0] for item in signal]
    #print(signal)
        
    plt.figure(figsize=(16,16))
    cmap = cm.get_cmap(name='tab20')
    map_ = Basemap(llcrnrlon=bbox[0], llcrnrlat=bbox[1], urcrnrlon=bbox[2], urcrnrlat=bbox[3], 
                   resolution='i', lat_0 = 0, lon_0 = 0)

    map_.drawmapboundary(fill_color='#d4dadc')
    map_.fillcontinents(color='#fafaf8', lake_color='#d4dadc')
    map_.drawcountries()
    
    for id_signal in np.unique(signal):
        id_groups = signal==id_signal
        x, y = map_(nodes.longitude.values[id_keep][id_groups], nodes.latitude.values[id_keep][id_groups])
        #s_centerd = (signal-np.min(signal))/(np.max(signal)-np.min(signal))
        sc = map_.scatter(x, y, c=cmap(id_signal/np.max(signal)), zorder=3, s=10, label=cls_groups[u-1][id_signal])
    #cbar = map_.colorbar(sc,location='bottom',pad="5%")
    plt.title(title)
    lgnd = plt.legend()
    
    for handle in lgnd.legendHandles:
        if isinstance(handle, coll.PathCollection):
            handle.set_sizes([50])

    plt.show()