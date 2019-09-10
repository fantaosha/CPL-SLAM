/** This class encapsulates an instance of the rank-restricted Riemannian form
 * of the semidefinite relaxation solved by CPL-SLAM.
 *
 * Copyright (C) 2018 - 2019 by Taosha Fan (taosha.fan@gmail.com)
 */

#pragma once

/** Use external matrix factorizations/linear solves provided by SuiteSparse
 * (SPQR and Cholmod) */

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/SPQRSupport>
#include <Eigen/Sparse>

#include "CPL-SLAM/CPL-SLAM_types.h"
#include "CPL-SLAM/CPL-SLAM_utils.h"
#include "CPL-SLAM/Oblique.h"
#include "CPL-SLAM/RelativePoseMeasurement.h"

namespace CPL_SLAM {

/** The type of the sparse Cholesky factorization to use in the computation of
 * the orthogonal projection operation */
typedef Eigen::CholmodDecomposition<ComplexSparseMatrix>
    ComplexSparseCholeskyFactorization;
typedef Eigen::CholmodDecomposition<RealSparseMatrix>
    RealSparseCholeskyFactorization;

/** The type of the QR decomposition to use in the computation of the orthogonal
 * projection operation */
typedef Eigen::SPQR<ComplexSparseMatrix> ComplexSparseQRFactorization;
typedef Eigen::SPQR<RealSparseMatrix> RealSparseQRFactorization;

/** The type of the incomplete Cholesky decomposition we will use for
 * preconditioning the conjugate gradient iterations in the RTR method */
typedef Eigen::IncompleteCholesky<Complex>
    ComplexIncompleteCholeskyFactorization;
typedef Eigen::IncompleteCholesky<Scalar> RealIncompleteCholeskyFactorization;

class CPL_SLAMProblem {
public:
  /// PROBLEM DATA

  /** The specific retraction of the CPL-SLAM problem used) */
  Retraction rect_;

  /** The specific formulation of the CPL-SLAM problem to be solved
(translation-implicit, translation-explicit, or robust) */
  Formulation form_;

  /** Number of poses and landmarks*/
  Eigen::Matrix<size_t, 2, 1> n_{0, 0};
  size_t size_ = 0;

  /** Number of measurements */
  Eigen::Matrix<size_t, 2, 1> m_{0, 0};

  /** Dimensional parameter d for the special Euclidean group SE(d) over which
   * this problem is defined */
  size_t d_ = 0;

  /** Relaxation rank */
  size_t r_ = 0;

  /** The oriented incidence matrix A encoding the underlying measurement
   * graph for this problem */
  RealSparseMatrix A_;

  /** The matrices B1, B2, and B3 defined in equation (69) of the CPL-SLAM tech
   * report */
  ComplexSparseMatrix B1_, B2_, B3_;

  /** The matrix M parameterizing the quadratic form appearing in the Explicit
   * form of the CPL-SLAM problem (Problem 2 in the CPL-SLAM tech report) */
  ComplexSparseMatrix M_;

  /** The rotational connection Laplacian for the special Euclidean
   * synchronization problem, cf. eq. 14 of the CPL-SLAM tech report.  Only
   * used in Implicit mode.*/
  ComplexSparseMatrix LGz_;

  /** The weighted reduced oriented incidence matrix Ared Omega^(1/2) (cf.
   * eq. 39 of the CPL-SLAM tech report).  Only used in Implicit mode. */
  RealSparseMatrix Ared_SqrtOmega_;

  /** The transpose of the above matrix; we cache this for computational
   * efficiency, since it's used frequently.  Only used in Implicit mode.
   */
  RealSparseMatrix SqrtOmega_AredT_;

  /** The weighted translational data matrix Omega^(1/2) T (cf. eqs. 22-24
   * of the CPL-SLAM tech report.  Only used in Implicit mode. */
  ComplexSparseMatrix SqrtOmega_T_;

  /** The transpose of the above matrix; we cache this for computational
   * efficiency, since it's used frequently.  Only used in Implicit mode. */
  ComplexSparseMatrix TT_SqrtOmega_;

  /** An Eigen sparse linear solver that encodes the Cholesky factor L used
   * in the computation of the orthogonal projection function (cf. eq. 39 of the
   * CPL-SLAM tech report) */
  RealSparseCholeskyFactorization L_;

  /** An Eigen sparse linear solver that encodes the QR factorization used in
   * the computation of the orthogonal projection function (cf. eq. 98 of the
   * CPL-SLAM tech report) */

  // When using Eigen::SPQR, the destructor causes a segfault if this variable
  // isn't explicitly initialized (i.e. not just default-constructed)
  ComplexSparseQRFactorization *QR_ = nullptr;

  /** A Boolean variable determining whether to use the Cholesky or QR
   * decompositions for computing the orthogonal projection */
  ProjectionFactorization projection_factorization_;

  /** The preconditioning strategy to use when running the Riemannian
   * trust-region algorithm */
  Preconditioner preconditioner_;

  /** Diagonal Jacobi preconditioner */
  ComplexDiagonalMatrix Jacobi_precon_;

  /** Incomplete Cholesky Preconditioner */
  ComplexIncompleteCholeskyFactorization *iChol_precon_ = nullptr;

  /** Tikhonov-regularized Cholesky Preconditioner */
  RealSparseCholeskyFactorization reg_Chol_precon_;

  /** Upper-bound on the admissible condition number of the regularized
   * approximate Hessian matrix used for Cholesky preconditioner */
  Scalar reg_Chol_precon_max_cond_;

  /** The underlying manifold in which the generalized orientations lie in the
  rank-restricted Riemannian optimization problem (Problem 9 in the CPL-SLAM
  tech
  report).*/
  Oblique OB_;

  /** The retraction map used in Riemannian optimization.*/
  ComplexMatrix (Oblique::*retract_)(const ComplexMatrix &,
                                     const ComplexMatrix &) const;

public:
  /// CONSTRUCTORS AND MUTATORS

  /** Default constructor; doesn't actually do anything */
  CPL_SLAMProblem() {}

  /** Basic constructor.  Here
   *
   * - measurements is a vector of relative pose measurements defining the
          pose-graph SLAM problem to be solved.
   * - formulation is an enum type specifying whether to solve the simplified
   *      form of the SDP relaxation (in which translational states have been
   *      eliminated) or the explicit form (in which the translational states
   *      are explicitly represented).
   * - projection_factorization is an enum type specifying the kind of matrix
   *      factorization to use when computing the action of the orthogonal
   *      projection operator Pi.  Only operative when solving the Simplified
   *      formulation of the special Euclidean synchronization problem
   *  - preconditioner is an enum type specifying the preconditioning strategy
   *      to employ
   */
  CPL_SLAMProblem(
      const measurements_t &measurements,
      const Formulation &formulation = Formulation::Simplified,
      const Retraction &retraction = Retraction::Projection,
      const ProjectionFactorization &projection_factorization =
          ProjectionFactorization::Cholesky,
      const Preconditioner &preconditioner = Preconditioner::IncompleteCholesky,
      Scalar reg_chol_precon_max_cond = 1e6);

  /** Set the maximum rank of the rank-restricted semidefinite relaxation */
  void set_relaxation_rank(size_t rank);

  /// ACCESSORS

  /** Returns the specific formulation of this CPL-SLAM problem */
  Retraction retraction() const { return rect_; }

  /** Returns the specific formulation of this CPL-SLAM problem */
  Formulation formulation() const { return form_; }

  /** Returns the type of matrix factorization used to compute the action of the
   * orthogonal projection operator Pi when solving a Simplified instance of the
   * special Euclidean synchronization problem */
  ProjectionFactorization projection_factorization() const {
    return projection_factorization_;
  }

  /** Returns the preconditioning strategy */
  Preconditioner preconditioner() const { return preconditioner_; }

  /** Returns the maximum admissible condition number for the regularized
   * Cholesky preconditioner */
  Scalar regularized_Cholesky_preconditioner_max_condition() const {
    return reg_Chol_precon_max_cond_;
  }

  /** Returns the number of poses appearing in this problem */
  size_t num_poses() const { return n_[0]; }

  /** Returns the number of poses appearing in this problem */
  size_t num_landmarks() const { return n_[1]; }

  /** Returns the number of measurements in this problem */
  Eigen::Matrix<size_t, 2, 1> num_measurements() const { return m_; }

  /** Returns the dimensional parameter d for the special Euclidean group SE(d)
   * over which this problem is defined */
  size_t dimension() const { return d_; }

  /** Returns the relaxation rank r of this problem */
  size_t relaxation_rank() const { return r_; }

  /** Returns the oriented incidence matrix A of the underlying measurement
   * graph over which this problem is defined */
  const RealSparseMatrix &oriented_incidence_matrix() const { return A_; }

  /** Returns the StiefelProduct manifold underlying this CPL-SLAM problem */
  const Oblique &Oblique_manifold() const { return OB_; }

  /// OPTIMIZATION AND GEOMETRY

  /** Given a matrix X, this function computes and returns the orthogonal
   *projection Pi * X */
  // We inline this function in order to take advantage of Eigen's ability
  // to optimize matrix expressions as compile time
  inline ComplexMatrix Pi_product(const ComplexMatrix &X) const {
    if (projection_factorization_ == ProjectionFactorization::Cholesky) {
      ComplexMatrix Res(X);
      return X - SqrtOmega_AredT_ * L_.solve(Ared_SqrtOmega_ * X);
    } else {
      ComplexMatrix PiX = X;
      for (size_t c = 0; c < X.cols(); c++) {
        // Eigen's SPQR support only supports solving with vectors(!) (i.e.
        // 1-column matrices)
        PiX.col(c) = X.col(c) - SqrtOmega_AredT_ * QR_->solve(X.col(c));
      }
      return PiX;
    }
  }

  /** This function computes and returns the product QX */
  // We inline this function in order to take advantage of Eigen's ability to
  // optimize matrix expressions as compile time
  inline ComplexMatrix Q_product(const ComplexMatrix &X) const {
    return LGz_ * X + TT_SqrtOmega_ * Pi_product(SqrtOmega_T_ * X);
  }

  /** Given a matrix Y, this function computes and returns the matrix product
  SY, where S is the matrix that determines the quadratic form defining the
  objective  F(Y) := tr(S * Y' * Y) for the CPL-SLAM problem.  More precisely:
  *
  * If formulation == Implicit, this returns Q * Y, where Q is as defined in
  equation (24) of the CPL-SLAM tech report.
  *
  * If formulation == Explicit, this returns M * Y, where M is as defined in
  equation (18) of the CPL-SLAM tech report. */
  ComplexMatrix data_matrix_product(const ComplexMatrix &Y) const;

  /** Given a matrix Y, this function computes and returns F(Y), the value of
   * the objective evaluated at X */
  Scalar evaluate_objective(const ComplexMatrix &Y) const;

  /** Given a matrix Y, this function computes and returns nabla F(Y), the
   * *Euclidean* gradient of F at Y. */
  ComplexMatrix Euclidean_gradient(const ComplexMatrix &Y) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem and
   * the *Euclidean* gradient nabla F(Y) at Y, this function computes and
   * returns the *Riemannian* gradient grad F(Y) of F at Y */
  ComplexMatrix Riemannian_gradient(const ComplexMatrix &Y,
                                    const ComplexMatrix &nablaF_Y) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem,
   * this
   * function computes and returns grad F(Y), the *Riemannian* gradient of F
   * at Y */
  ComplexMatrix Riemannian_gradient(const ComplexMatrix &Y) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem, the
   * *Euclidean* gradient nablaF_Y of F at Y, and a tangent vector dotY in
   * T_D(Y), the tangent space of the domain of the optimization problem at Y,
   * this function computes and returns Hess F(Y)[dotY], the action of the
   * Riemannian Hessian on dotY */
  ComplexMatrix
  Riemannian_Hessian_vector_product(const ComplexMatrix &Y,
                                    const ComplexMatrix &nablaF_Y,
                                    const ComplexMatrix &dotY) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem, and
   * a tangent vector dotY in T_D(Y), the tangent space of the domain of the
   * optimization problem at Y, this function computes and returns Hess
   * F(Y)[dotY], the action of the Riemannian Hessian on dotX */
  ComplexMatrix
  Riemannian_Hessian_vector_product(const ComplexMatrix &Y,
                                    const ComplexMatrix &dotY) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem, and
   * a tangent vector dotY in T_D(Y), this function applies the selected
   * preconditioning strategy to dotY */
  ComplexMatrix precondition(const ComplexMatrix &Y,
                             const ComplexMatrix &dotY) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem and
  a
  tangent vector dotY in T_Y(E), the tangent space of Y considered as a generic
  matrix, this function computes and returns the orthogonal projection of dotY
  onto T_D(Y), the tangent space of the domain D at Y*/
  ComplexMatrix tangent_space_projection(const ComplexMatrix &Y,
                                         const ComplexMatrix &dotY) const;

  /** Given a matrix Y in the domain D of the CPL-SLAM optimization problem and
   * a
   * tangent vector dotY in T_D(Y), this function returns the point Yplus in D
   * obtained by retracting along dotY */
  ComplexMatrix retract(const ComplexMatrix &Y,
                        const ComplexMatrix &dotY) const;

  /** Given a point Y in the domain D of the rank-r relaxation of the CPL-SLAM
   * optimization problem, this function computes and returns a matrix X = [t |
   * R] comprised of translations and rotations for a set of feasible poses for
   * the original estimation problem obtained by rounding the point Y */
  ComplexVector round_solution(const ComplexMatrix Y) const;

  /** Given a critical point Y of the rank-r relaxation of the CPL-SLAM
   * optimization problem, this function computes and returns the corresponding
   * Lagrange multiplier matrix Lambda */
  RealDiagonalMatrix compute_Lambda(const ComplexMatrix &Y) const;

  /** Given a critical point Y in the domain of the optimization problem, this
   *function computes the smallest eigenvalue lambda_min of S - Lambda and its
   *associated eigenvector v.  Returns a Boolean value indicating whether the
   *Lanczos method used to estimate the smallest eigenpair converged to
   *within the required tolerance. */
  bool compute_S_minus_Lambda_min_eig(
      const ComplexMatrix &Y, Scalar &min_eigenvalue,
      ComplexVector &min_eigenvector, size_t &num_iterations,
      size_t max_iterations = 10000,
      Scalar min_eigenvalue_nonnegativity_tolerance = 1e-5,
      size_t num_Lanczos_vectors = 20) const;

  /** Computes and returns the chordal initialization for the rank-restricted
   * semidefinite relaxation */
  ComplexMatrix chordal_initialization() const;

  /** Randomly samples a point in the domain for the rank-restricted
   * semidefinite relaxation */
  ComplexMatrix random_sample() const;

  ~CPL_SLAMProblem() {
    if (QR_)
      delete QR_;

    if (iChol_precon_)
      delete iChol_precon_;
  }

  /// COMPLEX MATRIX EIGENVALUE COMPUTATION

  struct SparseClpMatProd {
    const ComplexSparseMatrix mat_;

    // Number of rows and columns of the mat_
    int rows_;
    int cols_;

    SparseClpMatProd(const ComplexSparseMatrix &mat);

    int rows() const { return rows_; };
    int cols() const { return cols_; };

    void perform_op(const Scalar *x, Scalar *y) const;
  };

  /// MINIMUM EIGENVALUE COMPUTATIONS

  /** This is a lightweight struct used in conjunction with Spectra to compute
   *the minimum eigenvalue and eigenvector of S - Lambda(X); it has a single
   *nontrivial function, perform_op(x,y), that computes and returns the product
   *y = (S - Lambda + sigma*I) x */
  struct SMinusLambdaProdFunctor {
    const CPL_SLAMProblem *problem_;

    // Diagonal blocks of the matrix Lambda
    RealDiagonalMatrix Lambda_;
    int n_;

    // Number of rows and columns of the matrix B - Lambda
    int rows_;
    int cols_;

    // Dimensional parameter d of the special Euclidean group SE(d) over which
    // this synchronization problem is defined
    int dim_;
    Scalar sigma_;

    // Constructor
    SMinusLambdaProdFunctor(const CPL_SLAMProblem *prob, const ComplexMatrix &Y,
                            Scalar sigma = 0);

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // Matrix-vector multiplication operation
    void perform_op(const Scalar *x, Scalar *y) const;
  };
};
} // namespace CPL_SLAM
