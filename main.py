import numpy as np
import numpy.linalg
import logging


def svd(A: np.ndarray):
    '''
    Function which determines Singular Value Decomposition of A into U, E, and V. Also solves for condition number
    using spectral norms and inverse of A matrix.

    :param A:   np.ndarray      Matrix to operate SVD on
    :return:    [U: np.ndarray, E: np.ndarray, V.T: np.ndarray], condition number k, inverse of A
    '''
    AAT = A.T @ A
    # Eigenvalues and Eigenvectors
    e_vals, e_vecs = np.linalg.eig(AAT)
    invert_flag = any([val == 0 for val in e_vals])
    if invert_flag:
        logging.warning('Matrix contains singular value(s) equal to 0, meaning the inverse will not exist.')

    ind = np.argsort(e_vals)[::-1]

    V = np.array(list(map(np.real, e_vecs[ind])))

    # Singular values
    sings = np.array(list(map(np.real, np.sqrt(e_vals[ind]))))

    # Pad E according to dimensions of original A
    E = np.pad(eps := np.diag(sings), [(0, max(0, A.shape[0] - eps.shape[0])), (0, max(0, A.shape[-1] - eps.shape[-1]))], mode='constant')

    # Remove Singular values of 0 to allow calculations of SVD regardless
    sings = sings[sings != 0]
    inv_s = np.pad(eps := np.diag(list(map(lambda x: 1 / x, sings))), [(0, max(0, A.shape[0] - eps.shape[0])), (0, max(0, A.shape[-1] - eps.shape[-1]))], mode='constant').T
    U = A @ V @ inv_s

    # Condition Number
    k = sings.max() / sings.min()

    return [U, E, V.T], k, 0 if invert_flag else V @ inv_s @ U.T


def spring_mass(masses: list, spring_constants: list, g=-9.81):
    '''
    Function to run spring mass experiment to find displacement, elongation, and internal stress for a given set of
    input masses and spring constants at a given gravity/acceleration.

    :param masses:              np.array        1-D array containing masses (in Kg for Newton output)
    :param spring_constants:    np.array        1-D array containing spring constants for corresponding spring in system
    :param g:                   float           acceleration/gravity expressed in m/s^2 (for Newton output)
    :return:                    u: np.array of mass displacements,
                                e: np.array of spring elongations,
                                w: np.array of internal stresses
    '''
    n_fixed = len(spring_constants) - len(masses) + 1
    if n_fixed == 0:
        logging.error(f'\tNumber of fixed ends must be greater than 0 for A to be invertible.')
    elif n_fixed in [1, 2]:
        logging.info(f'\tSolving Ku=f system for {n_fixed} free ends...')
    else:
        logging.error('\tStrange number of fixed ends.')

    A = np.zeros((len(spring_constants), len(masses)))
    for i in range(len(masses)):
        A[i][i] = 1
    for i in range(1, len(spring_constants)):
        A[i][i-1] = -1
    A = np.array(A)

    C = np.diag(list(map(lambda x: 1 / x, spring_constants)))

    K = A.T @ C @ A
    f = np.array(masses) * g

    [U, E, VT], k, K_inv = svd(K)

    u = K_inv @ f
    e = A @ u
    w = C @ e

    return u, e, w


def main():
    A = np.array([[-3, 1], [6, -2], [6, -2]])
    [U, E, VT], k, K_inv = svd(A)

    u, e, w = spring_mass([3, 4, 2, 3, 1], [2, 1, 2, 2, 2])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
