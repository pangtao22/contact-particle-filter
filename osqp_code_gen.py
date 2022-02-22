import osqp
import numpy as np
import scipy.sparse as sparse

nd = 4
Q = np.random.randn(nd, nd)
Qs = sparse.csc_matrix(Q + Q.T + 10*np.eye(nd))
A = sparse.csc_matrix(np.eye(nd))
b = np.zeros(nd)

prob = osqp.OSQP()
prob.setup(Qs, b, A, np.zeros(nd), np.full(nd, np.inf), verbose=True)
prob.codegen("qp", parameters='matrices')