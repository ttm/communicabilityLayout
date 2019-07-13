import numpy as n, time as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
# http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap  # https://umap-learn.readthedocs.io/en/latest/
# try megaman

from aux import *
def decompose(dimred, dim):
    if dimred == 'MDS': # slowest!
        embedding = MDS(n_components=dim, n_init=__inits, max_iter=__iters, n_jobs=-1, dissimilarity=__dis)
    elif dimred == 'ISOMAP': # slow
        embedding = Isomap(n_neighbors=__nneigh, n_components=dim, n_jobs=-1)
    elif dimred == 'LLE': # slow-acceptable
        embedding = LocallyLinearEmbedding(n_neighbors=__nneigh, n_components=dim, n_jobs=-1)
    elif dimred == 'TSNE': # acceptable
        embedding = TSNE(n_components=dim, n_iter=__iters, metric='precomputed', learning_rate=__lrate, perplexity=__perplexity)
    elif dimred == 'UMAP': # fast
        embedding = umap.UMAP(n_neighbors=__nneigh, n_components=dim, metric=__dis, min_dist=0.1)
    elif dimred == 'PCA': # fastest!
        embedding = PCA(n_components=dim)
    else:
        raise ValueError('dimension reduction method not recognized')

    positions = embedding.fit_transform(An)
    return positions

def clust(clu, i):
    if clu == 'KM':
        calg = KMeans(n_clusters=i, n_init=100,n_jobs=-1)
    elif clu == 'AG':
        calg = AgglomerativeClustering(n_clusters=i)
    elif clu == 'SP':
        # calg = SpectralClustering(n_clusters=i, affinity='precomputed', n_jobs=-1)
        calg = SpectralClustering(n_clusters=i, n_jobs=-1)
    elif clu == 'AF':
        # calg = AffinityPropagation(n_clusters=i, affinity='precomputed', n_jobs=-1)
        calg = AffinityPropagation()
    else:
        raise ValueError('clustering algorithm not recognized')
    res = calg.fit(pC)
    return [int(j) for j in res.labels_]

##################### settings ###################
### communicability calculations:
__fname = '../data/dolphinsA.txt'
__fname = '../data/polblogs_A_cc.txt'
__fname = '../data/Benguela_A.txt'
__fname = '../data/YeastS_main.txt'
__mangle = 10e-5
__temp = 1

### dimensionality reduction:
__dis = 'euclidean'
__dis = 'precomputed'
# __dimred = 't-SNE'
__dimred = 'UMAP'
__dimred = 'ISOMAP'
__dimred = 'MDS'
__dimred = 'TSNE'
__dimred = 'PCA'  # implement, maybe also LDA or other methods
__dim = 3

__dimredC = 'TSNE'
__dimredC = 'MDS'
__dimredC = 'UMAP'
__dimredC = 'ISOMAP'
__dimredC = 'TSNE'
__dimredC = 'PCA'  # implement, maybe also LDA or other methods
__dimC = 5

__inits = 3  # for MDS ~3
__inits = 1  # for MDS ~3
__inits = 30  # for MDS ~3
__iters = 1000  # for MDS (~100) and t-SNE (~250)
__iters = 1000  # for MDS (~100) and t-SNE (~250)
__perplexity = 5  # for t-SNE
__lrate = 12  # for t-SNE
__nneigh = 300  # for UMAP

__cdmethod = 'an'  # angles
# __cdmethod = 'dist'  # remove this option dist

### for community detection:
__minclu = 2  # minimum number of clusters, not used for now
__nclu = 6  # maximum number of clusters, not used for now
__cddim = 'nd'  # the dimmensionality in which community detection is performed
__cddim = 'rd'  # the dimmensionality in which community detection is performed
__cddim = 3  # the dimmensionality in which community detection is performed
__clu = 'KM'
__clu = 'AG'
__clu = 'SP'
__clu = 'AF'

### plotting
plot = True


################## algorithm #########################
tt = t.time()
A = n.loadtxt(__fname)
As = n.maximum(A, A.T) - n.diag(n.diag(A))
N = As.shape[0]

G = expm(__temp*As)  # communicability matrix using Pade approximation
sc = n.matrix(n.diag(G)).T  # vector of self-communicabilities

u = n.matrix(n.ones(N)).T

if __cdmethod == 'dist':
    CD = n.dot(sc, u.T) + n.dot(u, sc.T) -2 * G  # squared communicability distance matrix
    CD[CD < 0] = 0
    X = n.array(CD) ** .5  # communicability distance matrix

c = G / (n.array(n.dot(sc, u.T)) * n.array( n.dot(u, sc.T))) ** .5
c[c > 1] = 1
An___ = n.arccos(c)
# An___ = n.arccos((G / (n.array(n.dot(sc, u.T)) * n.array( n.dot(u, sc.T)))) ** .5)
An__ = n.degrees(An___)
min_angle = __mangle
An_ = An__ + min_angle - n.identity(N) * min_angle
An = n.real( n.maximum(An_, An_.T) ) # communicability angles matrix

print('communcability calculations', t.time() - tt, 'net size', N)
tt = t.time()

# E_original = n.linalg.eigvals(An)

p = decompose(__dimred, __dim)
print('embedding', t.time() - tt)
tt = t.time()

p = .7 * p / n.abs(p).max()
if p.shape[1] == 3:
    sphere_data = getSphere(p)
else:
    sphere_data = getSphere(n.vstack((p.T, n.zeros(p.shape[0]))).T)
ll = n.vstack( A.nonzero() ).T.tolist()  # links
print('sphere', t.time() - tt)
tt = t.time()

# detecting communities
if __cddim == N:
    pC = An
elif (__dimredC == __dimred) and (__dimC == __dim):
    pC = p
else:
    pC = decompose(__dimredC, __dimC)
    print('second embedding', t.time() - tt)
    tt = t.time()

km = []
ev = []
if __minclu == 1:
    __minclu = 2
    ev.append(-5)
    km.append([0]*N)
nclusts = list(range(__minclu, __nclu +1 ))


# Hierarchical clustering Ward
# DBSCAN
# Spectral Clustering
if __clu in ('AF', 'DB'):
    labels = clust(__clu, 0)
    km.append(labels)
    ev.append(1)
else:
    for i in nclusts:
        # kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(p)
        labels = clust(__clu, i)
        km.append(labels)
        score = silhouette_score(pC, labels)
        ev.append(score)
print('clustering', t.time() - tt)
tt = t.time()

# return jsonify({
#     'nodes': p.tolist(), 'links': ll, 'sdata': sphere_data,
#     'ev': ev, 'clusts': km
# })

###################
# drawing
### matplotlib
best = nclusts[ev.index(max(ev))]
km_ = km[ev.index(max(ev))]
if plot:
    ll = n.vstack( A.nonzero() ).T  # links
    ll_ = ll.reshape(ll.shape[0]*ll.shape[1])
    plot3d(p, ll_, km_)
