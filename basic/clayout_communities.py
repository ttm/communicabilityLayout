import numpy as n

from scipy.linalg import expm
from sklearn.manifold import MDS

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt

################
# settings:
fname = '../data/ZackarA.txt'
fname = '../data/dolphinsA.txt'
fname = '../data/Roget_main_A.txt'
fname = '../data/YeastS_main.txt'
fname = '../data/neurons_A.txt'

# for communicability
beta = 1  # temperature
min_angle = 1e-5  # minimum communicability angle

# for MDS:
dims = 3
n_init = 3  # Number runs with different initialization
max_iter = 100  # max number of iterations in a single run
n_jobs = -1  # use all processors
# metrics = ['metricstress', 'metricsstress', 'sammon','strain'] dropped

# for community detection
ncluin = 2  # minimum number of clusters, not used for now
nclu = 6  # maximum number of clusters, not used for now


#########################
# procedures:
A = n.loadtxt(fname)
As = n.maximum(A, A.T) - n.diag( n.diag(A) )
N = As.shape[0]

print('calc G')
G = expm(beta*As)  # communicability matrix using Pade approximation
sc = n.matrix(n.diag(G)).T  # vector of self-communicabilities

u = n.matrix(n.ones(N)).T

# CD = n.dot(sc, u.T) + n.dot(u, sc.T) -2 * G  # squared communicability distance matrix
# CD_ = n.array(CD)
# neg = CD_ < 0
# X = n.array(CD) ** .5  # communicability distance matrix

print('calc An')
An___ = n.arccos(G / (n.array(n.dot(sc, u.T)) * n.array( n.dot(u, sc.T))) ** .5)
An__ = n.degrees(An___)
An_ = An__ + min_angle - n.identity(N) * min_angle
An = n.real( n.maximum(An_, An_.T) ) # communicability angles matrix

# E_original = n.linalg.eigvals(An)

print('calc MDS')
embedding = MDS(n_components=dims, n_init=n_init, max_iter=max_iter, n_jobs=n_jobs, dissimilarity='precomputed')

p = positions = embedding.fit_transform(An)

#########################
# community detection

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print('calc kmeans')
km = []
ev = []
nclusts = list(range(ncluin, nclu+1))
for i in nclusts:
    kmeans = KMeans(n_clusters=i, random_state=0).fit(An)
    km.append(kmeans)
    # score = silhouette_score(An, kmeans.labels_, metric='precomputed')
    score = silhouette_score(An, kmeans.labels_)
    ev.append(score)

best = nclusts[ev.index(max(ev))]
km_ = km[ev.index(max(ev))]

###################
# drawing
### matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print('plot kmeans')
for c in range(km_.n_clusters):
    yes = km_.labels_ == c
    pp = p[yes]
    ax.scatter(pp[:,0], pp[:,1], pp[:,2])

ll = n.vstack( A.nonzero() ).T.tolist()  # links
for l in ll: 
    p0 = p[l[0]]
    p1 = p[l[1]]
    lp = n.vstack(( p0, p1 ))
    ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], 'c')
plt.show()
