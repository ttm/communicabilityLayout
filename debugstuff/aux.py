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

    Note
    ----
        http://www.nervouscomputer.com/hfs/cmdscale-in-python/

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

import sympy
import numpy as np
def give_coords(distances):
    """give coordinates of points for which distances given

    coordinates are given relatively. 1st point on origin, 2nd on x-axis, 3rd 
    x-y plane and so on. Maximum n-1 dimentions for which n is the number
    of points

     Args:
        distanes (list): is a n x n, 2d array where distances[i][j] gives the distance 
            from i to j assumed distances[i][j] == distances[j][i]

     Returns:
        numpy.ndarray: cordinates in list form n dim

     Examples:
        >>> a = sympy.sqrt(2)
        >>> distances = [[0,1,1,1,1,1],
                         [1,0,a,a,a,a],
                         [1,a,0,a,a,a],
                         [1,a,a,0,a,a],
                         [1,a,a,a,0,a],
                         [1,a,a,a,a,0]]
        >>> give_coords(distances)
        array([[0, 0, 0, 0, 0],
               [1, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]], dtype=object)

        >>> give_coords([[0, 3, 4], [3, 0, 5], [4, 5, 0]])
        array([[0, 0],
        [3, 0],
        [0, 4]], dtype=object)        

    """
    distances = np.array(distances)

    n = len(distances)
    X = sympy.symarray('x', (n, n - 1))

    for row in range(n):
        X[row, row:] = [0] * (n - 1 - row)

    for point2 in range(1, n):

        expressions = []

        for point1 in range(point2):
            expression = np.sum((X[point1] - X[point2]) ** 2) 
            expression -= distances[point1,point2] ** 2
            expressions.append(expression)

        X[point2,:point2] = sympy.solve(expressions, list(X[point2,:point2]))[1]

    return X

from scipy import spatial
x = n.array([ [1,2,3], [5,6,7], [9,8,7] ])
d = spatial.distance_matrix(x, x)
def give_coords2(D):
    # https://math.stackexchange.com/questions/156161/finding-the-coordinates-of-points-from-distance-matrix/423898#423898
    N = D.shape[0]
    Dr1 = D[0,:]
    Dr1_ = n.repeat(Dr1[:, n.newaxis], N, axis=1).T
    Dc1 = D[:,0]
    Dc1_ = n.repeat(Dc1[:, n.newaxis], N, axis=1)
    M = (Dr1_**2 + Dc1_**2 - D**2) / 2
    eva, eve = n.linalg.eig(M)
    points = eve * (eva**0.5)
    return points, eva, eve

