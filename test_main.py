from main import svd, spring_mass
import numpy as np
import logging


def test_svd():
    As = [
        np.array([[-3, 1], [6, -2], [6, -2]]),
        np.array([[3, 2, 2], [2, 3, -2]]),
        np.array([[-3, 1], [6, -2], [6, -2]]),
        np.array([[1, 1, 1], [3, 1, 4], [6, 7, 0]]),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3]])
    ]
    logging.info('Showing custom and numpy SVDs back-to-back\n\n')
    for A in As:
        logging.info('=' * 45)
        logging.info(A)
        logging.info('=' * 45)
        [U, E, VT], k, K_inv = svd(A)
        U2, E2, V2 = np.linalg.svd(A)

        logging.info('\tU')
        logging.info(U)
        logging.info(U2)
        logging.info('_' * 45)
        logging.info('\tE')
        logging.info(E)
        logging.info(E2)
        logging.info('_' * 45)
        logging.info('\tVT')
        logging.info(VT)
        logging.info(V2)
        logging.info('_' * 45 + '\n')


def test_spring_mass():
    inputs = [
        (np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1, 1])),
        (np.array([1, 2, 3, 4, 5]), np.array([10, 10, 10, 10, 10]))
    ]

    for ins in inputs:
        logging.info('=' * 45)
        logging.info(f'\tRunning Spring-Mass System for masses: {ins[0]} and spring constants: {ins[1]}')
        logging.info('=' * 45)
        u, e, w = spring_mass(*ins)

        logging.info('\tDisplacements')
        logging.info(u)
        logging.info('_' * 45)

        logging.info('\tElongation')
        logging.info(e)
        logging.info('_' * 45)

        logging.info('\tStress')
        logging.info(w)
        logging.info('_' * 45 + '\n')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_svd()
    test_spring_mass()
