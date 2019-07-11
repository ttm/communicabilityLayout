import numpy as n
from scipy.linalg import expm
from sklearn.manifold import MDS, TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def getSphere(points):
    data = sphereFit(points[:,0], points[:,1], points[:,2])
    dists = (
                  (points[:,0] - data['c'][0])**2 
                + (points[:,1] - data['c'][1])**2 
                + (points[:,2] - data['c'][2])**2 
            ) ** 0.5
    mean = dists.mean()
    std = dists.std()
    return {**data, **{'mean': mean, 'std': std}}
def sphereFit(spX,spY,spZ):
    #   Assemble the A matrix
    spX = n.array(spX)
    spY = n.array(spY)
    spZ = n.array(spZ)
    A = n.zeros((len(spX),4))
    A[:,0] = spX*2
    A[:,1] = spY*2
    A[:,2] = spZ*2
    A[:,3] = 1

    #   Assemble the f matrix
    f = n.zeros((len(spX),1))
    f[:,0] = (spX*spX) + (spY*spY) + (spZ*spZ)
    C, residules, rank, singval = n.linalg.lstsq(A,f)

    #   solve for the radius
    t = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = t**0.5

    return {'r': radius[0], 'c': [C[0][0], C[1][0], C[2][0]]}

##################### settings ###################
### communicability calculations:
__fname = '../data/Benguela_A.txt'
__mangle = 10e-5
__temp = 1

### dimensionality reduction:
__dimred = 'MDS'
# __dimred = 't-SNE'
# __dimred = 'PCA'  # implement, maybe also LDA or other methods
__dim = 3
__inits = 30  # for MDS ~3
__iters = 1000  # for MDS (~100) and t-SNE (~250)
__perplexity = 5  # for t-SNE
__lrate = 12  # for t-SNE

__cdmethod = 'an'  # angles
# __cdmethod = 'dist'

### for community detection:
__minclu = 2  # minimum number of clusters, not used for now
__nclu = 6  # maximum number of clusters, not used for now
__cddim = 'rd'  # the dimmensionality in which community detection is performed

################## algorithm #########################
A = n.loadtxt(__fname)
As = n.maximum(A, A.T) - n.diag(n.diag(A))
N = As.shape[0]

G = expm(__temp*As)  # communicability matrix using Pade approximation
sc = n.matrix(n.diag(G)).T  # vector of self-communicabilities

u = n.matrix(n.ones(N)).T

if __cdmethod == 'dist':
    CD = n.dot(sc, u.T) + n.dot(u, sc.T) -2 * G  # squared communicability distance matrix
    X = n.array(CD) ** .5  # communicability distance matrix

c = G / (n.array(n.dot(sc, u.T)) * n.array( n.dot(u, sc.T))) ** .5
c[c > 1] = 1
An___ = n.arccos(c)
# An___ = n.arccos((G / (n.array(n.dot(sc, u.T)) * n.array( n.dot(u, sc.T)))) ** .5)
An__ = n.degrees(An___)
min_angle = __mangle
An_ = An__ + min_angle - n.identity(N) * min_angle
An = n.real( n.maximum(An_, An_.T) ) # communicability angles matrix

# E_original = n.linalg.eigvals(An)

if __dimred == 'MDS':
    embedding = MDS(n_components=__dim, n_init=__inits, max_iter=__iters, n_jobs=-1, dissimilarity='precomputed')
else:
    embedding = TSNE(n_components=__dim, n_iter=__iters, metric='precomputed', learning_rate=__lrate, perplexity=__perplexity)
p = positions = embedding.fit_transform(An)

p = .7 * p / n.abs(p).max()
if p.shape[1] == 3:
    sphere_data = getSphere(p)
else:
    sphere_data = getSphere(n.vstack((p.T, n.zeros(p.shape[0]))).T)
ll = n.vstack( A.nonzero() ).T.tolist()  # links

# detecting communities
km = []
ev = []

if __minclu == 1:
    __minclu = 2
    ev.append(-5)
    km.append([0]*N)
nclusts = list(range(__minclu, __nclu +1 ))

if __cdmethod == 'dist' and __cddim == 'rd':
    X_ = embedding.fit_transform(X)
for i in nclusts:
    # kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(p)
    if __cdmethod == 'an' and __cddim == 'nd':
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(An)
    if __cdmethod == 'an' and __cddim == 'rd':
        kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(p)
    if __cdmethod == 'dist':
        if __cddim == 'nd':
            kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(X)
        if __cddim == 'rd':
            kmeans = KMeans(n_clusters=i, n_init=100, max_iter=3000, n_jobs=-1, tol=1e-6).fit(X_)
    km.append([int(j) for j in kmeans.labels_])
    score = silhouette_score(p, kmeans.labels_)
    ev.append(score)

# return jsonify({
#     'nodes': p.tolist(), 'links': ll, 'sdata': sphere_data,
#     'ev': ev, 'clusts': km
# })
