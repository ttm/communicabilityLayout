import numpy as n
import vispy.scene
from vispy.scene import visuals
from IPython import get_ipython
ipython = get_ipython()
ipython.magic("gui qt5")

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

def plot3d(p, ll, km_):
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    for c in range(max(km_) + 1):
        yes = n.array(km_) == c
        pp = p[yes]
        scatter = visuals.Markers()
        scatter.set_data(pp, edge_color=None, face_color=(c%2, (c+1)%2, (c//2)%2, .5), size=5)
        view.add(scatter)
    line = visuals.Line(pos=p[ll], connect='segments', color=(1,1,1,0.4), method='gl')
    view.add(line)
    view.camera = 'turntable'  # or try 'arcball'
    vispy.app.run()

import numpy as np

def cmdscale(D):
    """
    Classical multidimensional scaling (MDS)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)

    return Y, evals
