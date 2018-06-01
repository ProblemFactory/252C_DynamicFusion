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
