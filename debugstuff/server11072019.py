import numpy as n, time as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.smacof.html
# from sklearn.manifold import MDS
from sklearn.manifold import smacof
# http://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap  # https://umap-learn.readthedocs.io/en/latest/
# try megaman

from aux import *

##################### settings ###################
### communicability calculations:
__fname = '../data/dolphinsA.txt'
__fname = '../data/polblogs_A_cc.txt'
__fname = '../data/Benguela_A.txt'
__fname = '../data/YeastS_main.txt'
__mangle = 10e-5
__temp = 1

### dimensionality reduction:
__dimred = 'ISOMAP'
__dimred = 'ICA'
# __dimred = 't-SNE'
__dis = 'euclidean'
__dis = 'precomputed'
__dimred = 'MDS'
__dimred = 'PCA'  # implement, maybe also LDA or other methods
__dimred = 'UMAP'

__dimredC = 'ISOMAP'
__dimredC = 'ICA'
__dimredC = 't-SNE'
__dimredC = 'MDS'
__dimredC = 'PCA'  # implement, maybe also LDA or other methods
__dimredC = 'UMAP'
__dimC = 3

__dim = 3
__inits = 3  # for MDS ~3
__inits = 1  # for MDS ~3
__inits = 30  # for MDS ~3
__iters = 1000  # for MDS (~100) and t-SNE (~250)
__iters = 1000  # for MDS (~100) and t-SNE (~250)
__perplexity = 5  # for t-SNE
__lrate = 12  # for t-SNE
__nneigh = 3000  # for UMAP

__cdmethod = 'an'  # angles
# __cdmethod = 'dist'  # remove this option dist

### for community detection:
__minclu = 2  # minimum number of clusters, not used for now
__nclu = 6  # maximum number of clusters, not used for now
__cddim = 'nd'  # the dimmensionality in which community detection is performed
__cddim = 'rd'  # the dimmensionality in which community detection is performed
__cddim = 3  # the dimmensionality in which community detection is performed

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

if __dimred == 'MDS':
    # embedding = MDS(n_components=__dim, n_init=__inits, max_iter=__iters, n_jobs=-1, dissimilarity=__dis)
    embedding = smacof(An)
    # foo = cmdscale(An)
    # p = foo[0][:,:3]
elif __dimred == 'PCA':
    embedding = PCA(n_components= __dim)
    p = positions = embedding.fit_transform(An)
elif __dimred == 'UMAP':
    embedding = umap.UMAP(n_neighbors=__nneigh, n_components=__dim, metric=__dis, min_dist=0)
    p = positions = embedding.fit_transform(An)
else:
    embedding = TSNE(n_components=__dim, n_iter=__iters, metric='precomputed', learning_rate=__lrate, perplexity=__perplexity)
    p = positions = embedding.fit_transform(An)
# p = positions = embedding.fit_transform(X)
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
km = []
ev = []
if __minclu == 1:
    __minclu = 2
    ev.append(-5)
    km.append([0]*N)
nclusts = list(range(__minclu, __nclu +1 ))

if __cddim == N:
    pC = An
elif (__dimredC == __dimred) and (__dimC == __dim):
    pC = p
else:
    pC = dimRed(An, __dimC, __dimredC)


for i in nclusts:
    # kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(p)
    if __cddim == N:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(An)
    elif __cddim == __dim:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(p)
    else:
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(X_)
    km.append([int(j) for j in kmeans.labels_])
    score = silhouette_score(p, kmeans.labels_)
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

# if plot:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
# 
#     print('plot kmeans')
#     for c in range(max(km_) + 1):
#         yes = n.array(km_) == c
#         pp = p[yes]
#         ax.scatter(pp[:,0], pp[:,1], pp[:,2])
# 
#     ll = n.vstack( A.nonzero() ).T.tolist()  # links
#     for l in ll: 
#         p0 = p[l[0]]
#         p1 = p[l[1]]
#         lp = n.vstack(( p0, p1 ))
#         ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], 'c')
#     print('plot set', t.time() - tt)
#     tt = t.time()
#     plt.show()
