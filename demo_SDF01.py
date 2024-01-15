'''
Signed distance function (SDF)

Sylvain Calinon, 2023
'''

import numpy as np
import matplotlib.pyplot as plt


def sdf_circle(x, p, r):
    d = np.linalg.norm(p-x) - r
    return d

def sdf_box(x, p, w, h):
    dtmp = np.abs(x-p) - np.array([w, h]) * 0.5
    d = np.linalg.norm(np.maximum(dtmp, [0, 0])) + np.min([np.max([dtmp[0], dtmp[1]]), 0])
    return d

def smooth_union(d1, d2, k):
    '''
    Smooth union (see https://www.shadertoy.com/view/lt3BW2)
    Note: will only be correct on the outside, see https://iquilezles.org/articles/interiordistance/
    '''
    h = np.max([k - np.abs(d1-d2), 0])
    d = np.min([d1, d2]) - (h**2)*0.25/k
    return d

def sdf(x):
    '''
    Compound shape 1
    '''
    p1 = np.array([0.36, 0.3])
    p2 = np.array([0.6, 0.5])
    d = np.zeros(x.shape[1])

    # Vectorize ?
    for t in range(x.shape[1]):
        d1 = sdf_circle(x[:,t], p1, 0.2)
        d2 = sdf_box(x[:,t], p2, 0.2, 0.4)
        d[t] = smooth_union(d1, d2, 0.1)

    return d

"""
Parameters
----------
"""
nbDim = 81;   # Number of datapoints per axis for visualization
nbIn = 2;       # Dimension of input data (here: 2D surface embedded in 3D)


"""
SDF Generation
--------------
"""
[X1, X2] = np.meshgrid(np.linspace(0,1,nbDim), np.linspace(0,1,nbDim));
x = np.vstack((X1.flatten(), X2.flatten()))
y = sdf(x);

# Numerical gradient computation
#e0 = 1E-6;
#dx = zeros(size(x,1), nbFcts^nbIn);
#for i=1:size(x,1)
#	e = zeros(size(x,1), 1);
#	e(i) = e0;
#	ytmp = sdf(x + repmat(e, 1, nbFcts^nbIn));
#	dx(i,:) = (y - ytmp) / e0;
#end

"""
Plots
-----
"""
fig, ax = plt.subplots(1)
ax.axis('off')
#colormap(np.tile(np.linspace(1,.4,64), [3, 1]));
#ax.plot_surface(X1, X2, np.reshape(y, (nbFcts, nbFcts))-np.max(y), cmap='viridis', edgecolor='k')
ax.contour(
    X1,
    X2,
    np.reshape(y, (nbDim, nbDim)),
    levels=[0],
    linewidths=4
)
ax.contour(
    X1,
    X2,
    np.reshape(y, (nbDim, nbDim)),
    levels=np.arange(-1, 1, 1e-2),
    linewidths=1
);
ax.axis('tight')
ax.axis('equal')
ax.invert_yaxis()
plt.show()

data = {
    'nbDim': nbDim,
    'x': x,
    'y': y,
}

np.save('sdf_obj02_81.npy', data, allow_pickle=True)
#print('-dpng','sdf_obj01.png');
