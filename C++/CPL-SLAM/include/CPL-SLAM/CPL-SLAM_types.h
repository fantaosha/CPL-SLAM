/** A set of typedefs describing the types of matrices and factorizations that
 * will be used in the CPL-SLAM algorithm.
 *
 * Copyright (C) 2018 - 2019 by Taosha Fan (taosha.fan@gmail.com)
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "Optimization/Smooth/TNT.h"

namespace CPL_SLAM {

/** Some useful typedefs for the CPL-SLAM library */
typedef double Scalar;
typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> RealVector;
typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> RealMatrix;
typedef Eigen::DiagonalMatrix<Complex, Eigen::Dynamic> ComplexDiagonalMatrix;
typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> RealDiagonalMatrix;

/** We use row-major storage order to take advantage of fast (sparse-matrix) *
 * (dense-vector) multiplications when OpenMP is available (cf. the Eigen
 * documentation page on "Eigen and Multithreading") */
typedef Eigen::SparseMatrix<Complex, Eigen::RowMajor> ComplexSparseMatrix;
typedef Eigen::SparseMatrix<Scalar, Eigen::RowMajor> RealSparseMatrix;

/** The specific formulation of special Euclidean synchronization problem to
 * solve */
enum class Formulation {
  /** Construct and solve the simplified version of the special Euclidean
   * synchronization problem obtained by analytically eliminating the
   *  translational states from the estimation (cf. Problem 4 in the CPL-SLAM
   * tech report).
   */
  Simplified,

  /** Construct and solve the formulation of the special Euclidean
   * synchronization problem that explicitly estimates both rotational and
   * translational states (cf. Problem 2 in the CPL-SLAM tech report).
   */
  Explicit
};

/** The type of factorization to use when computing the action of the orthogonal
 * projection operator Pi when solving the Simplified form of the special
 * Euclidean synchronization problem */
enum class ProjectionFactorization { Cholesky, QR };

/** The set of available preconditioning strategies to use in the Riemannian
 * Trust Region when solving this problem */
enum class Preconditioner {
  None,
  Jacobi,
  IncompleteCholesky,
  RegularizedCholesky
};

/** The strategy to use for constructing an initial iterate */
enum class Initialization { Chordal, Random };

/** The strategy to use for retraction */
enum class Retraction { Exponential, Projection };

/** A typedef for a user-definable function that can be used to
 * instrument/monitor the performance of the internal Riemannian
 * truncated-Newton trust-region optimization algorithm as it runs (see the
 * header file Optimization/Smooth/TNT.h for details). */
typedef Optimization::Smooth::TNTUserFunction<ComplexMatrix, ComplexMatrix,
                                              Scalar, ComplexMatrix>
    CPL_SLAMTNTUserFunction;

} // namespace CPL_SLAM
