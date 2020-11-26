#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:14:29 2020

@author: Renee
"""

# This script contains functions for implementing graph clustering and signal processing.


import numpy as np
import codecs
import json
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
import scipy.stats as stats
import numpy.random as random
from sklearn.cluster import KMeans


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


# BEGIN PS2 FUNCTIONS


def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
    
    gt = (np.arange(N)%k) # create uniformly partitioned ground truth
    random.shuffle(gt) # shuffle
    coords = np.empty([N,2])
    for i in np.arange(k): # create coords, sampling from norm with means spaced around unit circle
        coords[gt==i,0] = stats.norm(loc=np.cos((2*np.pi)/k*i), scale=sigma).rvs(size=np.sum(gt==i))
        coords[gt==i,1] = stats.norm(loc=np.sin((2*np.pi)/k*i), scale=sigma).rvs(size=np.sum(gt==i))
    
    unifsamp = np.reshape(stats.uniform().rvs(size=N*N),(N,N)) # uniform sample mat
    
    probs = np.zeros([N,N])
    for i in range(k):
        idx = np.where(gt==i)[0]
        probs[idx,:] = gt==i
    probs[probs==0] = pij #pij when not same cluster
    probs[probs==1] = pii #pii when same cluster
    A = np.array(unifsamp < probs).astype(int)
    gt = gt[:,np.newaxis] # adjust dimensions (easier if add dim after operations above)
    
    return A, gt, coords


def L(A, normalized=True):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    
    D = np.sum(A, axis=1) # here consider adjacency mat the weights W
    L = np.diag(D) - A # this is the combinatorial graph Laplacian
    
    if normalized:
        D_powered = np.power(D,-0.5,out=np.zeros(D.shape),where=(D!=0))
        L = np.diag(D_powered) @ L @ np.diag(D_powered) #formula for normalized
        
    return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    
    e, psi = eigh(L)
    e = e[:,np.newaxis]
    
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    
    s_hat = psi.T @ s
    
    return s_hat


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    
    H = psi @ np.diag(h(e)) @ psi.T
    
    return H


def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    
    # perform kmeans
    rep_labels = np.zeros([nrep,len(X)]) # labels generated by each repetition
    dist_to_cent = np.zeros(nrep) # within cluster distance for each rep
    for rep in range(nrep):
        old_labels = np.zeros(len(X))
        centroids = kmeans_plusplus(X, k) # start these as centroids
        for i in range(itermax): # iterate clusters and centroids
            # first get new clusters
            dists = np.array([np.linalg.norm(X - centroids[c,:],axis=1) for c in range(k)])
            labels = np.argmin(dists, axis=0)
            if np.sum(labels != old_labels) < len(X)*.01: continue # stop if cluster assignments barely change
            old_labels = labels
            # now calculate new centroids
            for c in range(k): 
                centroids[c] = np.mean(X[labels==c,:], axis=0) # calculate a new centroid
        rep_labels[rep,:] = labels
        dist_to_cent[rep] = np.linalg.norm(X - centroids[labels.astype(int)])
    labels = rep_labels[np.argmin(dist_to_cent),:]
        
    return labels


def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    
    centroids = np.zeros([k,X.shape[1]])*np.nan
    choice = np.random.choice(len(X))
    centroids[0,:] = X[choice,:]
    D = np.zeros((len(X),k))
    D[:,0] = np.linalg.norm(X - centroids[0,:],axis=1)
    probs = D[:,0] / np.sum(D[:,0]) # probs for first iteration
    for i in range(1,k):
        choice = np.random.choice(len(X), p=probs) # next centroid
        centroids[i,:] = X[choice,:]
        D[:,i] = np.linalg.norm(X - centroids[i,:],axis=1)
        probs = np.amin(D[:,:i+1],axis=1) / np.sum(np.amin(D[:,:i+1],axis=1))
    
    return centroids


def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
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
        _, psi = eigh(L)
        psi_k = psi[:,:k]
    else:  # just grab the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize your eigenvector rows
    psi_norm = psi_k / np.linalg.norm(psi_k, axis=1)[:,np.newaxis]

    if sklearn:
        labels = KMeans(n_clusters=k, n_init=nrep,
                        max_iter=itermax).fit_predict(psi_norm)
    else:
        labels = kmeans(psi_norm, k, nrep=nrep, itermax=itermax)

    return labels

