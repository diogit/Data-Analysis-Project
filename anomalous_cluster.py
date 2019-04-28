"""
Anomalous Cluster Algorithm
"""

import numpy as np

def anomalous(X, me, rang, D):
    """
    anomalous, iterative extraction of anomalous clusters based on the algorithm referred to as Separate/Conquer (Mirkin, 1999, Machine Learning)
    Input:
    X - data matrix,
    me - grand mean,
    range - normalizing values
    D - normalised data scatter
    output:
    ancl - an indexed dataset of anomalous clusters found:
    ancl[ind, 1] list of indices in cluster,
    ancl[ind, 2] standardised center,
    ancl[ind, 3] contribution to the data scatter
    """
    nn,mm = X.shape
    remains = np.array(range(nn))
    number = 0
    anomalous_cluster = []
    while len(remains) != 0:
        distance = dist(X,remains,rang,me)
        d, ind = max(distance), np.argmax(distance)
        index = remains[ind]
        centroid = X[index,:]
        number += 1
        # finding AP cluster
        cluster, centroid = anpat(X, remains, rang, centroid, me)
        ancl = [0,0,0]
        # list of indices in the anomalous cluster
        ancl[0] = cluster
        # standardized center of the anomalous cluster
        censtand = (centroid - me) / rang
        # cluster contrib, per cent
        dD = (np.dot(censtand, censtand.T) * len(cluster) *100) / D
        #print('[DEBUG] dD: ',dD)
        # standardised centroid
        ancl[1] = censtand
        # proportion of the data scatter
        ancl[2] = dD
        # update of the set of objects yet unclustered
        remains = np.setdiff1d(remains, cluster)
        anomalous_cluster.append(ancl)
    return anomalous_cluster

def anpat(X, remains, rang, centroid, me):
    """
    anpat, iterative soubroutine in anomalous
    based on the algorithm 'Separate/Conquer' (Mirkin, 1999, Machine Learning)
    Input:
    X - full data matrix
    remains - set of its row indices % %(objects) under consideration
    range - normalizing values
    centroid - initial center of the anomalous cluster
    me - vector to shift to the 0 (origin)
    Output:
    cluster - set of row indices in the anomalous cluster
    centrod - center of the cluster
    """
    count = 0
    key = 1
    while key == 1:
        # separc: separate cluster around "centroid" from that around "me"
        cluster = separc(X,remains,rang,centroid,me)
        if len(cluster) > 0:
            # center: finding centre of cluster
            newcenter = center(X,cluster)
        else:
            newcenter = centroid
        if not np.array_equal(newcenter, centroid):
            centroid = newcenter
        else:
            key = 0
    return cluster, centroid

def center(X, cluster):
    "Finding centroid to a cluster"
    nn, mm = X.shape
    ccc = np.zeros((mm))
    for jj in range(mm):
        zz = X[:,jj]
        zc = zz[cluster]
        ccc[jj] = zc.mean()
    return ccc

def separc(X0, remains, rang, a, b):
    "Separating a cluster around 'a' from that around 'b'"
    dist_a = distm(X0,remains,rang,a)
    dist_b = distm(X0,remains,rang,b)
    clust = dist_a < dist_b
    cluster = remains[clust]
    return cluster

def distm(X,remains,rang,a):
    """
    Finding normalized distances in 'remains' to point 'a'
    Input:
    X - the original data matrix
    remains - set of X-row indices under consideration
    range - normalizing vector
    a - point the distances relate to
    Output:
    distan - column of distances from a to remains
    """
    nn, mm = X.shape
    rr = len(remains)
    # Submatrix of the entities (remains)
    z = X[remains,:]
    # column vector with 'a' point
    az = np.array([a]*rr)
    rz = np.array([rang]*rr)
    # the relevant data normalized
    dz = (z - az) / rz
    # the squares of the relevant entries
    ddz = dz * dz
    # di is a row of the squared Euclidean distances
    if mm > 1:
        di = sum(ddz.T)
    else:
        di = ddz.T
    #rr times 1 column
    distan = di.T
    return distan

def dist(X, remains, stand, me):
    """
    Finding normalized distances in 'remains' to point 'a'
    Input:
    X - data matrix,
    remains - data still to cluster,
    stand - values to standardize,
    me - array point to calculate dist
    Output:
    distan - column of distances from a to remains.
    """
    nn, mm = X.shape
    rr = len(remains)
    # distan is a matrix rr x 1
    distan = np.zeros((rr))
    for jj in range(mm):
        # jjth feature
        z = X[:,jj]
        # the jjth feature of the data still in remains
        zz = z[remains]
        y = zz - me[jj]
        y = y / stand[jj]
        yy = y * y
        distan = distan + yy
    return distan

def post_process_ANCL(ancl, threshold=10):
    """
    post-processing of anomalous clusters
    removal of small anomalous clusters
    Input:
    ancl - anomalous clusters
    threshold - the maximum cardinality of a small cluster
    Output:
    new_ancl - new anomalous clusters
    """
    new_ancl = []
    aK = len(ancl)
    ll = np.zeros((aK))
    for ik in range(aK):
        ll[ik] = len(ancl[ik][0])
    rl = np.argwhere(ll>threshold)[:,0]
    # cent shape is (n_clusters x n_features)
    cent = np.zeros((len(rl),len(ancl[0][1])))

    if rl.size == 0:
        print('Too great a threshold!!! ')
    else:
        for ik in range(len(rl)):
            new_ancl.append(ancl[rl[ik]])
    return new_ancl
