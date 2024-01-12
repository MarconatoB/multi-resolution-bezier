import matplotlib.pyplot as plt
import numpy as np
import trimesh
import skimage
import io
import PIL.Image as Image
import tempfile

data = np.load('3D_rec/C0_iter3.npy', allow_pickle=True).item()
nbDim = data['nbDim']
sdf = data['y'].reshape((nbDim, nbDim, nbDim))
level_sets = [0,0.1,0.2,0.3,0.4,0.5]
scene = trimesh.Scene()
colors = plt.get_cmap('rainbow')(np.linspace(0, 1, len(level_sets)))
colors[1:,3] = 50 # set alpha
for i in range(len(level_sets)):
    vertices, faces, normals, _ = skimage.measure.marching_cubes(sdf, level=level_sets[i])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    mesh.visual.face_colors = colors[i]
    scene.add_geometry(mesh)

scene.show()
