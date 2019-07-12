# -*- coding: utf-8 -*-
# vispy: gallery 10
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

""" Demonstrates use of visual.Markers to create a point cloud with a
standard turntable camera to fly around with and a centered 3D Axis.
"""

import numpy as np, sys
import vispy.scene
from vispy.scene import visuals
# auto pypt5
# vispy.app.use_app(backend_name='pyqt4')

#
# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# generate data
pos = np.random.normal(size=(100, 3), scale=0.2)
pos2 = np.random.normal(size=(100, 3), scale=0.2) + 3
# one could stop here for the data generation, the rest is just to make the
# # data look more interesting. Copied over from magnify.py
# centers = np.random.normal(size=(50, 3))
# indexes = np.random.normal(size=100000, loc=centers.shape[0]/2.,
#                            scale=centers.shape[0]/3.)
# indexes = np.clip(indexes, 0, centers.shape[0]-1).astype(int)
# scales = 10**(np.linspace(-2, 0.5, centers.shape[0]))[indexes][:, np.newaxis]
# pos *= scales
# pos += centers[indexes]

# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(pos, edge_color=None, face_color=(1, 1, 1, .5), size=5)
scatter2 = visuals.Markers()
scatter2.set_data(pos2, edge_color=None, face_color=(1, 0, 1, .5), size=5)

line = visuals.Line(pos = np.array([[2,4,5], [3,7,1], [0,0,0], [2,2,2], [2,0,0], [4,2,2]]), connect='segments', color=(1,1,0,1), method='gl')

view.add(scatter)
view.add(scatter2)
view.add(line)

view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)

if __name__ == '__main__':
    print('banana')
    vispy.app.run()
    # import sys
    # if sys.flags.interactive != 1:
    #     vispy.app.run()
