from scipy.spatial import distance_matrix
import numpy as np

def getNodes(vertices, radius, n_nodes=np.inf):
    nodes = np.array([], dtype=np.int)
    idces = np.arange(vertices.shape[0])
    while idces.shape[0]>0 and nodes.shape[0]<n_nodes:
        idx = np.random.choice(idces, size=(1,))
        D = distance_matrix(vertices[idx,:], vertices[idces,:])<=radius
        nodes = np.append(nodes, [idx])
        idces = idces[~np.any(D, axis=0)]
    return nodes

def k_nearest(verts, nodes, k):
    l = 0
    if verts is nodes:
        l = 1
    result = np.zeros((verts.shape[0], k), dtype=np.int)
    result_D = np.zeros((verts.shape[0], k))
    max_per_slice = int(np.floor(1e7/nodes.shape[0])) #about 800 MB memory
    n_slices = int(np.ceil(verts.shape[0]/max_per_slice))
    for i in range(n_slices):
        cur_slice_length = min(verts.shape[0]-i*max_per_slice, max_per_slice)
        D = distance_matrix(verts[i*max_per_slice:i*max_per_slice+cur_slice_length], nodes)
        result[i*max_per_slice:i*max_per_slice+cur_slice_length, :] = np.argsort(D, axis=1)[:, l:l+k]
        result_D[i*max_per_slice:i*max_per_slice+cur_slice_length, :] = D[np.arange(cur_slice_length, dtype=np.int)[:,np.newaxis], result[i*max_per_slice:i*max_per_slice+cur_slice_length, :]]
    return result, result_D