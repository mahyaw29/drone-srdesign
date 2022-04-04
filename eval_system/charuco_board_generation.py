# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 16:32:36 2022

@author: Mahya
"""

#%%
import numpy as np
import cv2, PIL, os
from cv2 import aruco
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
%matplotlib nbagg

#%%
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)

fig = plt.figure()
nx = 8
ny = 6
for i in range(1, nx*ny+1):
    ax = fig.add_subplot(nx, ny, i)
    img = aruco.drawMarker(aruco_dict, i-1, 700)
    plt.imshow(img, cmap = mpl.cm.gray, interpolation="nearest")
    ax.axis("off")

plt.savefig("images/markers.pdf")
plt.show()

board = aruco.CharucoBoard_create(10, 7, 1, 0.8, aruco_dict)
imboard = board.draw((4000, 4000))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
cv2.imwrite("images/chessboard.tiff", imboard)

plt.grid()
plt.show()