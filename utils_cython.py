import numpy as np


def GetUpperTriangularDataAlongColumn(Q):
    """
    Convert Q into csc format (up->down and then left->right.)
    :param Q: 2D numpy array representing a square matrix.
    :return:
    """
    nd = Q.shape[0]
    Qnew = np.zeros(int(nd*(nd+1)/2))
    start = 0
    for i in range(nd):
        Qnew[start:start+(i+1)] = Q[0:(i+1), i]
        start += i+1

    return Qnew


def GetRotationMatrixFromNormal(normal):
    R = np.eye(3)
    R[:, 2] = normal
    if np.linalg.norm(normal[:2]) < 1e-6:
        R[:, 0] = [0, normal[2], -normal[1]]
    else:
        R[:, 0] = [normal[1], -normal[0], 0]
    R[:, 0] /= np.linalg.norm(R[:, 0])
    R[:, 1] = np.cross(normal, R[:, 0])

    return R


def CalcTangentVectors(normal, nd):
    normal = normal.copy()
    normal /= np.linalg.norm(normal)
    if nd == 2:
        # Makes sure that dC is in the yz plane.
        dC = np.zeros((2, 3))
        dC[0] = np.cross(np.array([1, 0, 0]), normal)
        dC[1] = -dC[0]
    else:
        R = GetRotationMatrixFromNormal(normal)
        dC = np.zeros((nd, 3))

        for i in range(nd):
            theta = 2 * np.pi / nd * i
            dC[i] = [np.cos(theta), np.sin(theta), 0]

        dC = (R.dot(dC.T)).T
    return dC


def CalcFrictionConeRays(normal, mu, nd):
    """
    number of extreme rays of the polydedron friction cone.
    :param normal:
    :param mu:
    :param nd:
    :return:
    """
    dC = CalcTangentVectors(normal, nd)
    vC = mu * dC + normal
    vC /= np.sqrt(1 + mu ** 2)
    return vC.T


def skew_symmetric(w: np.array) -> np.array:
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0., -w[0]],
                  [-w[1], w[0], 0]])
    return W
