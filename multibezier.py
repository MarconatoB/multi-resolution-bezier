import math
import os
import numpy as np
import matplotlib.pyplot as plt

"""
Utility functions
-----------------
"""
def binomial(n, i):
    if n >= 0 and i >= 0:
         b = math.factorial(n) / (math.factorial(i) * math.factorial(n - i))
    else:
        b = 0
    return b


def block_diag(A,B):
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
    out[:A.shape[0], :A.shape[1]] = A
    out[A.shape[0]:, A.shape[1]:] = B
    return out


def cubic_bezier_basis(nbDim, nbSeg, edges_at_zero=False, C1=True):
    """
    Computes the matrix phi whose columns are cubic Bernstein polynomials

    Parameters
    ----------
    nbDim : int
        desired length of each basis function (number of rows of phi)
    nbSeg : int
        number of segments to concatenate
    edges_at_zero : bool
        if true, the extremities will be constrained to 0
    C1 : bool
        if true,  the resulting curve will be C1 continuous
        if false, the resulting curve will be C0 continuous

    Returns
    -------
    phi : basis functions matrix
    C : constraint matrix, useful to retrieve control points

    """
    nbT = nbDim // nbSeg
    nbFct = 4
    t = np.linspace(0, 1 - 1 / nbT, nbT)

    T0 = np.zeros((len(t), nbFct))

    for n in range(nbFct):
        T0[:, n] = np.power(t, n)

    B0 = np.zeros((nbFct, nbFct))
    for n in range(1, nbFct + 1):
        for i in range(1, nbFct + 1):
            tmp = (-1) ** (nbFct - i - n)
            tmp *= -binomial(nbFct - 1, i - 1)
            tmp *= binomial(nbFct - 1 - (i - 1), nbFct - 1 - (n - 1) - (i - 1))
            B0[nbFct - i, n - 1] = tmp
    T = np.kron(np.eye(nbSeg), T0)
    B = np.kron(np.eye(nbSeg), B0)

    if C1 is True: # Continuous derivative constraint
        C0 = np.array([[1, 0, 0, -1], [0, 1, 1, 2]]).T
        if nbFct >= 4:
            C0 = block_diag(np.eye(nbFct-4),C0)

        C = np.eye(2)
        for n in range(nbSeg-1):
            C = block_diag(C,C0)

        C = block_diag(C,np.eye(nbFct-2))

        if edges_at_zero is True:
            # remove the 2 first and 2 last columns to force keypoints to 0
            C = C[:,2:-2]

    else: # C0 continuity constraint
        C0 = np.array([[1], [1]])
        C0 = block_diag(C0, np.eye(2))

        C = np.eye(3)
        for n in range(nbSeg-1):
            C = block_diag(C,C0)

        C = block_diag(C,np.eye(1))

        if edges_at_zero is True:
            # remove the first and last columns to force keypoints to 0
            C = C[:,1:-1]

    phi = T @ B @ C

    return phi, C


# TODO: re-implement the C matrix computation for any nbSeg
def quad_bezier_basis(nbDim, nbSeg, edges_at_zero=False, C1=True):
    """
    Computes the matrix phi whose columns are quadratic Bernstein polynomials

    Parameters
    ----------
    nbDim : int
        desired length of each basis function (number of rows of phi)
    nbSeg : int
        number of segments to concatenate
    edges_at_zero : bool
        if true, the extremities will be constrained to 0
    C1 : bool
        if true,  the resulting curve will be C1 continuous
        if false, the resulting curve will be C0 continuous

    Returns
    -------
    phi : basis functions matrix
    C : constraint matrix, useful to retrieve control points

    """
    nbT = nbDim // nbSeg
    nbFct = 3
    t = np.linspace(0, 1 - 1 / nbT, nbT)

    T0 = np.zeros((len(t), nbFct))

    for n in range(nbFct):
        T0[:, n] = np.power(t, n)

    B0 = np.zeros((nbFct, nbFct))
    for n in range(1, nbFct + 1):
        for i in range(1, nbFct + 1):
            tmp = (-1) ** (nbFct - i - n)
            tmp *= -binomial(nbFct - 1, i - 1)
            tmp *= binomial(nbFct - 1 - (i - 1), nbFct - 1 - (n - 1) - (i - 1))
            B0[nbFct - i, n - 1] = tmp
    T = np.kron(np.eye(nbSeg), T0)
    B = np.kron(np.eye(nbSeg), B0)

    if C1 is True: # Continuous derivative constraint
        # ! Only valid for nbSeg = 2
        C = np.array([
            [1,    0, 0, 0,   0],
            [0,    1, 0, 0,   0],
            [0,    0, 1, 0,   0],
            [0,    0, 1, 0,   0],
            [0,   -1, 2, 0,   0],
            [0, -1/2, 1, 1/2, 0],
            [0, -1/2, 1, 1/2, 0],
            [0,    0, 0, 1,   0],
            [0,    0, 0, 0,   1]
        ])
        if edges_at_zero is True:
            # remove the 2 first and 2 last columns to force keypoints to 0
            C = C[:,2:-2]

    else: # C0 continuity constraint
        # ! Only valid for nbSeg = 2
        C0 = np.array([
            [1, 0],
            [1, 0]
        ])
        C = np.eye(2)
        for n in range(nbSeg-1):
            C = block_diag(C,C0)
            C = block_diag(C, np.eye(1))

        C = block_diag(C,np.eye(1))

        if edges_at_zero is True:
            # remove the first and last columns to force keypoints to 0
            C = C[:,1:-1]

    phi = T @ B @ C

    return phi, C


def subdivide_2D(x):

    axis0_center = x.shape[0] // 2
    axis1_center = x.shape[1] // 2

    subdivisions = [
        x[:axis0_center, :axis1_center],
        x[:axis0_center, axis1_center:],
        x[axis0_center:, :axis1_center],
        x[axis0_center:, axis1_center:]
    ]
    return subdivisions


def subdivide_3D(x):

    axis0_center = x.shape[0] // 2
    axis1_center = x.shape[1] // 2
    axis2_center = x.shape[2] // 2

    subdivisions = [
        x[:axis0_center, :axis1_center, :axis2_center],
        x[:axis0_center, :axis1_center, axis2_center:],
        x[:axis0_center, axis1_center:, :axis2_center],
        x[:axis0_center, axis1_center:, axis2_center:],

        x[axis0_center:, :axis1_center, :axis2_center],
        x[axis0_center:, :axis1_center, axis2_center:],
        x[axis0_center:, axis1_center:, :axis2_center],
        x[axis0_center:, axis1_center:, axis2_center:],
    ]
    return subdivisions

"""
Plotting functions
------------------
"""
def plot_1D(Psi, weights, x0, subdivisions=[], layers_indices = [], title=""):
    """
    Plot each reconstruction layer (contribution of each iteration)
    """

    fig, axs = plt.subplots(len(layers_indices)+1, 1, sharex=True)

    if len(layers_indices) > 0:
        ax = axs[-1]
        fig.set_size_inches(8,2*(len(layers_indices)+1))

        for l, layer in enumerate(layers_indices):
            x_layer = Psi[:, layer]@weights[layer]
            axs[l].plot(x_layer)
            axs[l].set_title(f"Layer {l+1}")
            for s, segment in enumerate(subdivisions):
                axs[l].axvline(segment[0], c='black', alpha=0.5)
    else:
        ax = axs
        fig.set_size_inches(8, 6)


    ax.plot(Psi@weights, label="reconstruction")
    ax.plot(x, label="original")
    for s, segment in enumerate(subdivisions):
        ax.axvline(segment[0], c='black', alpha=0.5)
    ax.legend()
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()

    return


def plot_2D(x, in_shape, subdivisions=[], title=""):
    T1, T2 = np.meshgrid(
        np.linspace(0, 1, in_shape[0]),
        np.linspace(0, 1, in_shape[1])
    )

    plt.figure(figsize=(8, 8))

    # Reconstructed surface
    plt.axis('off')
    # ax2.plot_surface(T1, T2, np.reshape(xb, (nbDim, nbDim)) - np.max(xb), cmap='viridis', edgecolor='k')
    plt.contour(T1, T2, np.reshape(x, in_shape), levels=np.arange(0, 1, 0.02), linewidths=2)
    msh = plt.contour(T1, T2, np.reshape(x, in_shape), levels=[0], linewidths=4, colors='b')
    plt.axis('tight')
    plt.axis('equal')
    plt.title(title)

    # Subdivisions
    nbDim = in_shape[0] # assuming square input
    for segment in subdivisions:
        x = (segment[0,0] % nbDim)/nbDim
        y = (segment[0,0] // nbDim)/nbDim
        width = segment.shape[0]/nbDim
        height = segment.shape[1]/nbDim
        p = plt.gca().add_patch(plt.Rectangle((x,y), width, height, fill=False))

    return

"""
Fitting algorithm
-----------------
"""
def adaptive_fit(x, in_shape, nbOut, nbFct, iterations=1, threshold=0.0, C1=True):
    """
    Computes an approximation of a given signal using a superposition of concatenated
    Bézier curves of different resolutions by recursive subdivisions of the signal

    Parameters
    ----------
    x : signal to fit (can be 1D, 2D or 3D)
    in_shape : tuple with the size of each input dimension
    nbOut: number of output dimensions
    nbFct: degree + 1 of polynomials used for the fitting
    iterations: number of subdivisions to perform
    threshold: max RMSE of each segment for the subdivision criterion
    C1 : bool, whether the concatenated polynomials should be C1 continuous

    Returns
    -------
    x = Psi @ weights
    Psi : basis functions matrix
    weights : weights of each basis polynomial
    """
    if nbFct == 4:  # cubic
        nbSeg = 2
        suffix='cubic_'
    elif nbFct == 3:
        nbSeg = 3
        suffix='quad_'

    if C1 is True:
        suffix += 'C1'
    else:
        suffix += 'C0'

    print(f"Fitting a {in_shape} -> ({nbOut}) signal with {suffix} curves")
    # Make directories to store plots and 3D approximations
    os.makedirs("figures", exist_ok=True)
    os.makedirs("3D_rec", exist_ok=True)

    residual = np.copy(x)
    Psi = np.array([]).reshape(x.size, 0)
    weights = np.array([])
    subdivisions = [np.arange(0, x.size).reshape(in_shape)]  # indices
    layers_indices = []     # to keep track of each layer contribution

    for i in range(0, iterations):
        new_subdivisions = []

        for region in subdivisions:
            residual_error = np.linalg.norm(residual[region])   # Frobenius norm
            if residual_error > threshold:
                # Build a 1D basis
                if nbFct == 3:
                    phi_local,_ = quad_bezier_basis(
                        region.shape[0],
                        nbSeg,
                        edges_at_zero = (i>0),
                        C1=C1)
                elif nbFct == 4:
                    phi_local,_ = cubic_bezier_basis(
                        region.shape[0],
                        nbSeg,
                        edges_at_zero = (i>0),
                        C1=C1)

                # Extend 1D basis to multiple dimensions
                Psi_local = np.zeros(
                    (x.size, phi_local.shape[1]**len(in_shape))
                )
                psi_tmp = np.copy(phi_local)
                for _ in range(len(in_shape)-1):
                    psi_tmp = np.kron(psi_tmp, phi_local)

                Psi_local[region.flatten()] = np.kron(psi_tmp, np.eye(nbOut))

                # Augment previous basis with new detail functions
                Psi = np.hstack((Psi, Psi_local))

                if len(in_shape) == 1:
                    new_subdivisions.extend(np.array_split(region, nbSeg))
                elif len(in_shape) == 2:
                    new_subdivisions.extend(subdivide_2D(region))
                elif len(in_shape) == 3:
                    new_subdivisions.extend(subdivide_3D(region))

            else:
                new_subdivisions.append(region)

        # Recompute coefficients
        weights = np.linalg.pinv(Psi)@x
        residual = x - Psi@weights
        total_RMSE = np.linalg.norm(residual)

        subdivisions = new_subdivisions

        if i > 0:
            layers_indices.append(
                np.arange(layers_indices[i-1][-1]+1, len(weights))
            )
        else:
            layers_indices = [np.arange(len(weights))]

        # Visualization
        title = f"{i+1} iterations, threshold={threshold:.0e}, RMSE={total_RMSE:.2e}, {len(weights)} weights, {suffix}"
        if len(in_shape) == 1:
            plot_1D(Psi, weights, x0, subdivisions, [], title)
            filename = "figures/1D_" + suffix + f"_iter{i+1}"
            plt.savefig(filename)
            plt.show()
            plt.close()
        elif len(in_shape) == 2:
            plot_2D(Psi@weights, in_shape, subdivisions, title)
            filename = "figures/2D_" + suffix + f"_iter{i+1}"
            plt.savefig(filename)
            plt.show()
            plt.close()
        elif len(in_shape) == 3:
            data = {
                'nbDim': nbDim,
                'x': np.arange(x.size).reshape((in_shape)),
                'y': Psi@weights,
            }
            label = 'C1' if C1 is True else 'C0'
            np.save(f"3D_rec/{label}_iter{i+1}.npy", data)

        print(f"{i+1} iterations, {len(subdivisions)} subdivisions, RMSE={total_RMSE:.2e}, {len(weights)} weights")

    return Psi, weights




"""
1D test
-------
"""
"""
For quadratic curves, use a SDF with 3^n samples
so that the region can be split into 3 segments each iteration
"""
data = np.load('sdf_obj02_81.npy',allow_pickle=True).item() # reference
nbDim = data['nbDim']
t12 =  data['x']
x0 = np.copy(data['y'])

nbIn = t12.shape[0]
in_shape = tuple([nbDim for i in range(nbIn)])
nbOut = 1

# Slice of the SDF as a 1D test signal
x = x0[round(60/128*nbDim)::nbDim]

# Using quadratic C1 continuous curves
Psi, w = adaptive_fit(
    x,
    tuple([nbDim]),
    nbOut,
    nbFct=3,
    iterations=4,
    threshold=1e-4,
    C1=True
)


# Using quadratic C0 continuous curves
Psi, w = adaptive_fit(
    x,
    tuple([nbDim]),
    nbOut,
    nbFct=3,
    iterations=4,
    threshold=1e-4,
    C1=False
)


"""
For cubic curves, use a SDF with 2^n samples
so that the region can be split into 2 segments each iteration
"""
data = np.load('sdf_obj02.npy',allow_pickle=True).item()    # reference
nbDim = data['nbDim']
t12 =  data['x']
x0 = np.copy(data['y'])

nbIn = t12.shape[0]
in_shape = tuple([nbDim for i in range(nbIn)])
nbOut = 1


# Slice of the SDF as a 1D test signal
x = x0[round(60/128*nbDim)::nbDim]

# Using cubic C1 continuous curves
Psi, w = adaptive_fit(
    x,
    tuple([nbDim]),
    nbOut,
    nbFct=4,
    iterations=5,
    threshold=1e-4,
    C1=True
)

# Using cubic C0 continuous curves
Psi, w = adaptive_fit(
    x,
    tuple([nbDim]),
    nbOut,
    nbFct=4,
    iterations=5,
    threshold=1e-4,
    C1=False
)

"""
2D test
-------
"""
plot_2D(x0, in_shape, title=f"Original SDF with dimensions {in_shape}")
plt.savefig("figures/2D_original")
plt.close()
# Using cubic C1 continuous curves
Psi, w = adaptive_fit(
    x0,
    in_shape,
    nbOut,
    nbFct=4,
    iterations=5,
    threshold=1e-3,
    C1=True
)

# Using cubic C0 continuous curves
Psi, w = adaptive_fit(
    x0,
    in_shape,
    nbOut,
    nbFct=4,
    iterations=5,
    threshold=1e-3,
    C1=False
)

"""
3D SDF test (slow)
------------------
Each iteration's reconstruction is stored in another file for external visualization
"""
data = np.load('../robotics-codes-from-scratch-master/data/sdf3D01.npy',allow_pickle=True).item()
nbDim = data['nbDim']
t12 = data['x']   # 3D
y = data['y']

nbIn = t12.shape[0]
in_shape = tuple([nbDim for i in range(nbIn)])
nbOut = 1

x0 = np.copy(y)

Psi, weights = adaptive_fit(
    x0.flatten(),
    in_shape,
    nbOut,
    nbFct=4,
    iterations=5,
    threshold=1e-2,
    C1=True
)

Psi, weights = adaptive_fit(
    x0.flatten(),
    in_shape,
    nbOut,
    nbFct=4,
    iterations=4,
    threshold=1e-2,
    C1=False
)
