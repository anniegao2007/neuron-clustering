import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

def cmeans(X, k, m=2.0, ep=0.01, nrep=5, itermax=300):
    best_distance = float('inf')
    best_weights = []
    best_centroids = []
    num_pts = X.shape[0]
    for _ in range(nrep):
        # centroids = np.random.uniform(low=0.3, high=0.7, size=(k, X.shape[1]))  # randomly initialize weights
        centroids = kmeans_plusplus(X, k) + np.random.normal(scale=0.2, size=(k, X.shape[1]))
        old_weights = np.zeros((num_pts,k))
        for _ in range(itermax):
            dist_to_centroids = np.array([np.linalg.norm(X[x] - centroids, axis=1) for x in range(num_pts)])
            weights = np.empty((num_pts,k))
            for i in range(num_pts):
                for j in range(k):
                    weights[i,j] = 1.0 / (np.sum([(dist_to_centroids[i,j] / dist_to_centroids[i,k]) ** 2 for k in range(k)]))
            if np.linalg.norm(weights - old_weights) <= ep:  # think about this later
                # print('early stopping!')
                break
            for c in range(k):
                centroids[c] = np.sum(X * weights[:,c][:,None], axis=0) / np.sum(weights[:,c])
            old_weights = weights
        within_cluster_dist = np.sum([weights[:,c] * np.linalg.norm(X - centroids[c]) for c in range(k)])
        if within_cluster_dist < best_distance:
            best_distance = within_cluster_dist
            best_weights = weights
            best_centroids = centroids
    return best_weights, best_centroids  # best_labels


def kmeans(X, k, thresh=0.45, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    # best_distance = float('inf')
    # best_labels = []
    num_pts = X.shape[0]
    # for _ in range(nrep):
    centroids = kmeans_plusplus(X, k) + np.random.normal(scale=0.2, size=(k, X.shape[1]))  # find your initial centroids
    # print(centroids)
    old_labels = np.zeros((num_pts,k))
    for _ in range(itermax):
        dist_to_centroids = np.array([np.linalg.norm(X[x] - centroids, axis=1) for x in range(X.shape[0])])
        dist_to_centroids = np.reshape(np.sum(dist_to_centroids, axis=1), (num_pts, 1)) / dist_to_centroids
        dist_to_centroids /= np.reshape(np.sum(dist_to_centroids, axis=1), (num_pts, 1))
        # todo: sample based on the probability, jittering kmeans or kmeans++
        labels = np.where(dist_to_centroids > thresh, 1, 0)
        if np.linalg.norm(labels - old_labels) == 0:  # think about this later
            break
        for c in range(centroids.shape[0]):
            pts = np.where(labels[:,c])
            centroids[c] = np.mean(X[pts], axis=0)
        old_labels = labels
        # within_cluster_dist = np.linalg.norm(X - centroids[labels])
        # if within_cluster_dist < best_distance:
        #     best_distance = within_cluster_dist
        #     best_labels = labels
    return labels, centroids  # best_labels

def kmeans_vanilla(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    best_distance = float('inf')
    best_labels = []
    best_centroids = None
    for _ in range(nrep):
        centroids = kmeans_plusplus(X, k)  # find your initial centroids
        old_labels = np.zeros((X.shape[0],1))
        for _ in range(itermax):
            dist_to_centroids = np.array([np.linalg.norm(X[x] - centroids, axis=1) for x in range(X.shape[0])])
            labels = np.argmin(dist_to_centroids, axis=1)
            if np.linalg.norm(labels - old_labels) == 0:
                break
            for c in range(centroids.shape[0]):
                pts = np.where(labels == c)
                centroids[c] = np.mean(X[pts], axis=0)
            old_labels = labels
        within_cluster_dist = np.linalg.norm(X - centroids[labels])
        if within_cluster_dist < best_distance:
            best_distance = within_cluster_dist
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids

def cmeans(X, k, m=2, itermax=300):
    num_pts, num_dims = X.shape;
    coefs = np.random.random(size=[num_pts,k]) # random initial coefficients
    centroids = np.empty((k, num_dims))  # find initial centroids
    for j in range(centroids.shape[0]):
        centroids[j] = np.sum(coefs[:,j][:,np.newaxis]*X,axis=0)/np.sum(coefs[:,j])
    old_weights = np.zeros((num_pts,k))
    for _ in range(itermax):
        xc = np.array([np.linalg.norm(X-centroids[i],axis=1) for i in range(k)]).T
        um = 1/(np.array([np.nansum(np.divide(np.tile(xc[:,j],(k,1)).T,xc)**(2/(m-1)),axis=1) \
                          for j in range(k)]).T)
        if np.linalg.norm(um - old_weights) == 0:
            break
        for j in range(centroids.shape[0]):
            centroids[j] = np.sum(um[:,j][:,np.newaxis]*X,axis=0)/np.sum(um[:,j])
        old_weights = um
    return um, centroids  # best_labels

def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    centroids = np.empty((k, X.shape[1]))
    used = set()
    for i in range(k):
        rand_pt = np.random.random_integers(0, X.shape[0]-1)
        pts = [pt for pt in range(X.shape[0]) if pt not in used]
        dists = np.array([np.linalg.norm(X[pt,...] - X[rand_pt,...]) for pt in pts])

        # sum_dist = np.sum(dists)
        # print(rand_pt, dists)

        dists /= np.sum(dists)
        new_centroid_id = np.random.choice(pts, p=dists)
        centroids[i,...] = X[new_centroid_id,...]
        used.add(new_centroid_id)
    return centroids

# def kmeans_plusplus(X, k):
#     """kmeans_plusplus: initialization algorithm for kmeans
#     Args:
#         X (n x d np.ndarray): input data, rows = points, cols = dimensions
#         k (int): Number of clusters to partition

#     Returns:
#         centroids (k x d np.ndarray): centroids for initializing k-means
#     """
    
#     centroids = np.zeros([k,X.shape[1]])*np.nan
#     choice = np.random.choice(len(X))
#     centroids[0,:] = X[choice,:]
#     D = np.zeros((len(X),k))
#     D[:,0] = np.linalg.norm(X - centroids[0,:],axis=1)
#     probs = D[:,0] / np.sum(D[:,0]) # probs for first iteration
#     for i in range(1,k):
#         choice = np.random.choice(len(X), p=probs) # next centroid
#         centroids[i,:] = X[choice,:]
#         D[:,i] = np.linalg.norm(X - centroids[i,:],axis=1)
#         probs = np.amin(D[:,:i+1],axis=1) / np.sum(np.amin(D[:,:i+1],axis=1))
    
#     return centroids


def gaussian_kernel(X, kernel_type="gaussian", sigma=3.0, k=5):
    """gaussian_kernel: Build an adjacency matrix for data using a Gaussian kernel
    Args:
        X (N x d np.ndarray): Input data
        kernel_type: "gaussian" or "adaptive". Controls bandwidth
        sigma (float): Scalar kernel bandwidth
        k (integer): nearest neighbor kernel bandwidth
    Returns:
        W (N x N np.ndarray): Weight/adjacency matrix induced from X
    """
    _g = "gaussian"
    _a = "adaptive"

    kernel_type = kernel_type.lower()
    D = squareform(pdist(X))
    if kernel_type == "gaussian":  # gaussian bandwidth checking
        print("fixed bandwidth specified")

        if not all([type(sigma) is float, sigma > 0]):  # [float, positive]
            print("invalid gaussian bandwidth, using sigma = max(min(D)) as bandwidth")
            D_find = D + np.eye(np.size(D, 1)) * 1e15
            sigma = np.max(np.min(D_find, 1))
            del D_find
        sigma = np.ones(np.size(D, 1)) * sigma
    elif kernel_type == "adaptive":  # adaptive bandwidth
        print("adaptive bandwidth specified")

        # [integer, positive, less than the total samples]
        if not all([type(k) is int, k > 0, k < np.size(D, 1)]):
            print("invalid adaptive bandwidth, using k=5 as bandwidth")
            k = 5

        knnDST = np.sort(D, axis=1)  # sorted neighbor distances
        sigma = knnDST[:, k]  # k-nn neighbor. 0 is self.
        del knnDST
    else:
        raise ValueError

    W = ((D**2) / sigma[:, np.newaxis]**2).T
    W = np.exp(-1 * (W))
    W = (W + W.T) / 2  # symmetrize
    W = W - np.eye(W.shape[0])  # remove the diagonal
    return W

def L(A, normalized=False):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    D = np.diag(np.sum(A, axis=1))
    Lc = D - A
    if not normalized:
        return Lc
    d_neg = np.diag(np.sum(A, axis=1) ** -0.5)
    return d_neg @ Lc @ d_neg

# def L(A, normalized=True):
#     """L: compute a graph laplacian

#     Args:
#         A (N x N np.ndarray): Adjacency matrix of graph
#         normalized (bool, optional): Normalized or combinatorial Laplacian

#     Returns:
#         L (N x N np.ndarray): graph Laplacian
#     """
    
#     D = np.sum(A, axis=1) # here consider adjacency mat the weights W
#     L = np.diag(D) - A # this is the combinatorial graph Laplacian
    
#     if normalized:
#         D_powered = np.power(D,-0.5,out=np.zeros(D.shape),where=(D!=0))
#         L = np.diag(D_powered) @ L @ np.diag(D_powered) #formula for normalized
        
#     return L


def SC(L, k, alg=cmeans, thresh=None, psi=None, itermax=300):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # compute the first k elements of the Fourier basis
        # use scipy.linalg.eigh
        _, psi = np.linalg.eigh(L)
    # just grab the first k eigenvectors
    psi_k = psi[:, :k]

    # normalize your eigenvector rows
    psi_k /= np.linalg.norm(psi_k, axis=1)[:,None]
    if thresh is None:
        labels = alg(psi_k, k, itermax=itermax)
    else:
        labels = alg(psi_k, k, thresh=thresh, itermax=itermax)
    return labels