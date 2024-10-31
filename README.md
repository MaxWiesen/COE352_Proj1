# Project #1: SVD and Spring-Mass Systems
Written to satisfy the requirements of COE 352 Project #1, this project contains a `main.py` file containing an implementation of Singular Value Decomposition and a Ku = f Spring-Mass System solver as well as a `test_main.py` file to test those functions.

## SVD Function
The Singular Value Decomposition in `main.py` follows a fairly routine SVD algorithm. First, A<sup>T</sup>A is computed, from which `np.linalg.eig` is used to obtain the eigenvalues and eigenvectors. The eigenvectors are assembled into our V matrix, while the square roots of the eigenvalues are taken to determine our singular values. Our E matrix is then formed by diagonalizing our singular values in descending order. Finally, U can be found using A, V, and the inverse of the singular matrix (E). Before returning U, E, and V<sup>T</sup>, condition number k is computed using the spectral norms of the singular values. A<sup>-1</sup> is also calculated at this point for this implementation of SVD.

### This SVD vs `numpy.linalg.svd`
By running the SVD section of `test_main.py` file, we can gain insights into the differences in behavior between this implementation of SVD and that which is included in the `numpy.linalg` library. Structurally, small differences include that this SVD also returns the condition number and matrix inverse, while `numpy.linalg.svd` only returns the UEV<sup>T</sup>. Additionally, while E is returned as a singular matrix in this implementation, it is returned as an array of eigenvalues in `numpy`'s implementation. 

Analytically, the functions return virtually the same matrices, only numpy's implementation is better conditioned for precision errors, where this implementation occasionally leaves small artifacts (near 0) in the return matrices. This could easily be changed by adding a tolerance parameter.

## Spring-Mass System
The Spring-Mass System represents an experiment for singly or doubly fixed spring-mass systems, Ku = f, capable of solving equilibrium displacements from the force balance given a set of masses and a set of springs with their respective constants. After finding the displacement, elongations and internal stresses are obtained. First, a K matrix is populated using the `masses` and `spring_constant` list arguments. Next, the force array is assembled using the masses and input acceleration/gravity. Finally, using the components output by executing SVD on K, Ku = f can be inverted to solve for displacements u (from which elongation and internal stress can be found).

While the function will intake any set of masses and springs, it is only defined for one or two fixed ends. When the inputs reflect two free ends, the system is underdetermined, containing n variables with n-1 unknowns--since each fixed end represents a boundary condition. Alternatively, when there are more than two fixed ends, the Ku = f system, as we have defined it, will fail.
