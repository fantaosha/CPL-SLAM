#include "Spectra/MatOp/SparseSymMatProd.h"
#include "Spectra/SymEigsSolver.h"  // Spectra's symmetric eigensolver

#include "CPL-SLAM/CPL-SLAMProblem.h"
#include "CPL-SLAM/CPL-SLAM_utils.h"

#include <random>

namespace CPL_SLAM {

CPL_SLAMProblem::CPL_SLAMProblem(
    const measurements_t &measurements, const Formulation &formulation,
    const Retraction &retraction,
    const ProjectionFactorization &projection_factorization,
    const Preconditioner &precon, Scalar reg_chol_precon_max_cond)
    : rect_(retraction),
      form_(formulation),
      projection_factorization_(projection_factorization),
      preconditioner_(precon),
      reg_Chol_precon_max_cond_(reg_chol_precon_max_cond) {
  // Construct oriented incidence matrix for the underlying pose graph
  A_ = construct_oriented_incidence_matrix(measurements);

  // Construct B matrices (used to compute chordal initializations)
  construct_B_matrices(measurements, B1_, B2_, B3_);

  /// Set dimensions of the problem
  n_[0] = measurements.num_poses;
  n_[1] = measurements.num_landmarks;
  size_ = n_.sum();
  m_[0] = measurements.first.size();
  m_[1] = measurements.second.size();
  d_ = !(measurements.first.empty());
  r_ = d_;

  /// Set dimensions of the product of Stiefel manifolds in which the
  /// (generalized) rotational states lie
  OB_.set_n(n_[0]);
  OB_.set_p(r_);

  retract_ = (retraction == Retraction::Exponential) ? &Oblique::exp
                                                     : &Oblique::retract;

  if (form_ == Formulation::Simplified) {
    /// Construct data matrices required for the implicit formulation of the
    /// CPL-SLAM problem

    // Construct rotational connection Laplacian
    LGz_ = construct_rotational_connection_Laplacian(measurements);

    // Construct square root of the (diagonal) matrix of translational
    // measurement precisions
    RealDiagonalMatrix SqrtOmega =
        construct_landmark_and_translational_precision_matrix(measurements)
            .diagonal()
            .cwiseSqrt()
            .asDiagonal();

    // Construct Ared * SqrtOmega
    Ared_SqrtOmega_ = A_.topRows(size_ - 1) * SqrtOmega;

    // We cache the transpose of the above matrix as well to avoid having to
    // dynamically recompute this as an intermediate step each time the
    // transpose operator is applied
    SqrtOmega_AredT_ = Ared_SqrtOmega_.transpose();

    // Construct translational data matrix T
    ComplexSparseMatrix T =
        construct_landmark_and_translational_data_matrix(measurements);

    SqrtOmega_T_ = SqrtOmega * T;
    // Likewise, we also cache this transpose
    TT_SqrtOmega_ = SqrtOmega_T_.adjoint();

    /// Construct matrices necessary to compute orthogonal projection onto the
    /// kernel of the weighted reduced oriented incidence matrix Ared_SqrtOmega
    if (projection_factorization_ == ProjectionFactorization::Cholesky) {
      // Compute and cache the Cholesky factor L of Ared * Omega * Ared^T
      L_.compute(Ared_SqrtOmega_ * SqrtOmega_AredT_);
    } else {
      // Compute the QR decomposition of Omega^(1/2) * Ared^T (cf. eq. (98) of
      // the tech report).Note that Eigen's sparse QR factorization can only be
      // called on matrices stored in compressed format
      SqrtOmega_AredT_.makeCompressed();

      QR_ = new ComplexSparseQRFactorization();
      QR_->compute(SqrtOmega_AredT_.cast<Complex>());
    }

    /** Compute and cache preconditioning matrices, if required */
    if (preconditioner_ == Preconditioner::Jacobi) {
      ComplexVector diag = LGz_.diagonal();
      Jacobi_precon_ = diag.cwiseInverse().asDiagonal();
    } else if (preconditioner_ == Preconditioner::IncompleteCholesky)
      iChol_precon_ = new ComplexIncompleteCholeskyFactorization(LGz_);
    else if (preconditioner_ == Preconditioner::RegularizedCholesky) {
      // Compute maximum eigenvalue of LGz

      // NB: Spectra's built-in SparseSymProduct matrix assumes that input
      // matrices are stored in COLUMN-MAJOR order

      if (n_[1] == 0) {
        RealSparseMatrix LGz(2 * n_[0], 2 * n_[0]);
        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(4 * LGz_.nonZeros());

        for (int k = 0; k < LGz_.outerSize(); k++) {
          for (ComplexSparseMatrix::InnerIterator it(LGz_, k); it; ++it) {
            const auto &i = it.row();
            const auto &j = it.col();
            Scalar const &real = it.value().real();
            Scalar const &imag = it.value().imag();

            triplets.emplace_back(2 * i, 2 * j, real);
            triplets.emplace_back(2 * i, 2 * j + 1, -imag);
            triplets.emplace_back(2 * i + 1, 2 * j, imag);
            triplets.emplace_back(2 * i + 1, 2 * j + 1, real);
          }
        }

        LGz.setFromTriplets(triplets.begin(), triplets.end());

        SparseClpMatProd op(LGz_);
        Spectra::SymEigsSolver<Scalar, Spectra::LARGEST_MAGN, SparseClpMatProd>
            max_eig_solver(&op, 1, 3);
        max_eig_solver.init();

        int max_iterations = 10000;
        Scalar tol =
            1e-4;  // We only require a relatively loose estimate here ...
        int nconv = max_eig_solver.compute(max_iterations, tol);

        Scalar lambda_max = max_eig_solver.eigenvalues()(0);
        reg_Chol_precon_.compute(
            LGz + RealSparseMatrix(
                      RealVector::Constant(
                          LGz.rows(), lambda_max / reg_Chol_precon_max_cond_)
                          .asDiagonal()));
      } else {
        M_ = construct_quadratic_form_data_matrix(measurements);
        RealSparseMatrix M(4 * n_[0] + 2 * n_[1], 4 * n_[0] + 2 * n_[1]);

        std::vector<Eigen::Triplet<Scalar>> triplets;
        triplets.reserve(4 * M_.nonZeros());

        for (int k = 0; k < M_.outerSize(); k++) {
          for (ComplexSparseMatrix::InnerIterator it(M_, k); it; ++it) {
            auto const &i = it.row();
            auto const &j = it.col();
            Scalar const &real = it.value().real();
            Scalar const &imag = it.value().imag();

            triplets.emplace_back(2 * i, 2 * j, real);
            triplets.emplace_back(2 * i, 2 * j + 1, -imag);
            triplets.emplace_back(2 * i + 1, 2 * j, imag);
            triplets.emplace_back(2 * i + 1, 2 * j + 1, real);
          }
        }

        M.setFromTriplets(triplets.begin(), triplets.end());

        SparseClpMatProd op(M_);
        Spectra::SymEigsSolver<Scalar, Spectra::LARGEST_MAGN, SparseClpMatProd>
            max_eig_solver(&op, 1, 20);
        max_eig_solver.init();

        int max_iterations = 10000;
        Scalar tol =
            1e-4;  // We only require a relatively loose estimate here ...
        int nconv = max_eig_solver.compute(max_iterations, tol);

        Scalar lambda_max = max_eig_solver.eigenvalues()(0);
        reg_Chol_precon_.compute(
            M + RealSparseMatrix(
                    RealVector::Constant(M.rows(),
                                         lambda_max / reg_Chol_precon_max_cond_)
                        .asDiagonal()));
      }
    }
  } else {
    // form == Explicit
    M_ = construct_quadratic_form_data_matrix(measurements);

    /** Compute and cache preconditioning matrices, if required */
    if (preconditioner_ == Preconditioner::Jacobi) {
      ComplexVector diag = M_.diagonal();
      Jacobi_precon_ = diag.cwiseInverse().asDiagonal();
    } else if (preconditioner_ == Preconditioner::IncompleteCholesky)
      iChol_precon_ = new ComplexIncompleteCholeskyFactorization(M_);
    else if (preconditioner_ == Preconditioner::RegularizedCholesky) {
      // Compute maximum eigenvalue of M

      // NB: Spectra's built-in SparseSymProduct matrix assumes that input
      // matrices are stored in COLUMN-MAJOR order
      RealSparseMatrix M(4 * n_[0] + 2 * n_[1], 4 * n_[0] + 2 * n_[1]);

      std::vector<Eigen::Triplet<Scalar>> triplets;
      triplets.reserve(4 * M_.nonZeros());

      for (int k = 0; k < M_.outerSize(); k++) {
        for (ComplexSparseMatrix::InnerIterator it(M_, k); it; ++it) {
          auto const &i = it.row();
          auto const &j = it.col();
          Scalar const &real = it.value().real();
          Scalar const &imag = it.value().imag();

          triplets.emplace_back(2 * i, 2 * j, real);
          triplets.emplace_back(2 * i, 2 * j + 1, -imag);
          triplets.emplace_back(2 * i + 1, 2 * j, imag);
          triplets.emplace_back(2 * i + 1, 2 * j + 1, real);
        }
      }

      M.setFromTriplets(triplets.begin(), triplets.end());

      SparseClpMatProd op(M_);
      Spectra::SymEigsSolver<Scalar, Spectra::LARGEST_MAGN, SparseClpMatProd>
          max_eig_solver(&op, 1, 20);
      max_eig_solver.init();

      int max_iterations = 10000;
      Scalar tol =
          1e-4;  // We only require a relatively loose estimate here ...
      int nconv = max_eig_solver.compute(max_iterations, tol);

      Scalar lambda_max = max_eig_solver.eigenvalues()(0);
      reg_Chol_precon_.compute(
          M +
          RealSparseMatrix(RealVector::Constant(
                               M.rows(), lambda_max / reg_Chol_precon_max_cond_)
                               .asDiagonal()));
    }
  }
}

void CPL_SLAMProblem::set_relaxation_rank(size_t rank) {
  r_ = rank;
  OB_.set_p(r_);
}

ComplexMatrix CPL_SLAMProblem::data_matrix_product(
    const ComplexMatrix &Y) const {
  if (form_ == Formulation::Simplified)
    return Q_product(Y);
  else
    return M_ * Y;
}

Scalar CPL_SLAMProblem::evaluate_objective(const ComplexMatrix &Y) const {
  size_t size = 2 * Y.size();

  if (form_ == Formulation::Simplified) {
    ComplexMatrix QY = Q_product(Y);
    Eigen::Map<RealVector> v0((Scalar *)Y.data(), size);
    Eigen::Map<RealVector> v1((Scalar *)QY.data(), size);

    return v0.dot(v1);
  } else  // form == Explicit
  {
    ComplexMatrix MY = M_ * Y;
    Eigen::Map<RealVector> v0((Scalar *)Y.data(), size);
    Eigen::Map<RealVector> v1((Scalar *)MY.data(), size);

    return v0.dot(v1);
  }
}

ComplexMatrix CPL_SLAMProblem::Euclidean_gradient(const ComplexMatrix &Y) const {
  if (form_ == Formulation::Simplified)
    return 2 * data_matrix_product(Y);
  else  // form == Explicit
    return 2 * M_ * Y;
}

ComplexMatrix CPL_SLAMProblem::Riemannian_gradient(
    const ComplexMatrix &Y, const ComplexMatrix &nablaF_Y) const {
  return tangent_space_projection(Y, nablaF_Y);
}

ComplexMatrix CPL_SLAMProblem::Riemannian_gradient(
    const ComplexMatrix &Y) const {
  return tangent_space_projection(Y, Euclidean_gradient(Y));
}

ComplexMatrix CPL_SLAMProblem::Riemannian_Hessian_vector_product(
    const ComplexMatrix &Y, const ComplexMatrix &nablaF_Y,
    const ComplexMatrix &dotY) const {
  if (form_ == Formulation::Simplified)
    return OB_.Proj(Y,
                    2 * Q_product(dotY) - OB_.DDiagProduct(nablaF_Y, Y, dotY));
  else {
    // Euclidean Hessian-vector product
    ComplexMatrix H_dotY = 2 * M_ * dotY;

    H_dotY.block(size_, 0, n_[0], r_) =
        OB_.Proj(Y.block(size_, 0, n_[0], r_),
                 H_dotY.block(size_, 0, n_[0], r_) -
                     OB_.DDiagProduct(nablaF_Y.block(size_, 0, n_[0], r_),
                                      Y.block(size_, 0, n_[0], r_),
                                      dotY.block(size_, 0, n_[0], r_)));

    return H_dotY;
  }
}

ComplexMatrix CPL_SLAMProblem::Riemannian_Hessian_vector_product(
    const ComplexMatrix &Y, const ComplexMatrix &dotY) const {
  return Riemannian_Hessian_vector_product(Y, Euclidean_gradient(Y), dotY);
}

ComplexMatrix CPL_SLAMProblem::precondition(const ComplexMatrix &Y,
                                           const ComplexMatrix &dotY) const {
  if (preconditioner_ == Preconditioner::None)
    return dotY;
  else if (preconditioner_ == Preconditioner::Jacobi)
    return tangent_space_projection(Y, Jacobi_precon_ * dotY);
  else if (preconditioner_ == Preconditioner::IncompleteCholesky)
    return tangent_space_projection(Y, iChol_precon_->solve(dotY));
  else if (!n_[1]) {
    ComplexMatrix X(dotY.rows(), dotY.cols());
    Eigen::Map<RealMatrix> x((Scalar *)X.data(), 2 * dotY.rows(), dotY.cols());
    Eigen::Map<RealMatrix> y((Scalar *)dotY.data(), 2 * dotY.rows(),
                             dotY.cols());
    x = reg_Chol_precon_.solve(y);

    return tangent_space_projection(Y, X);
  } else {
    ComplexMatrix X(2 * n_[0] + n_[1], r_);
    ComplexMatrix Z(2 * n_[0] + n_[1], r_);
    Z.topRows(size_).setZero();
    Z.bottomRows(n_[0]) = dotY;
    Eigen::Map<RealMatrix> x((Scalar *)X.data(), 2 * X.rows(), X.cols());
    Eigen::Map<RealMatrix> z((Scalar *)Z.data(), 2 * Z.rows(), Z.cols());
    x = reg_Chol_precon_.solve(z);

    return tangent_space_projection(Y, X.bottomRows(n_[0]));
  }
}

ComplexMatrix CPL_SLAMProblem::tangent_space_projection(
    const ComplexMatrix &Y, const ComplexMatrix &dotY) const {
  if (form_ == Formulation::Simplified)
    return OB_.Proj(Y, dotY);
  else {
    // form == Explicit
    ComplexMatrix P(dotY.rows(), dotY.cols());

    // Projection of translational states is the identity
    P.block(0, 0, size_, r_) = dotY.block(0, 0, size_, r_);

    // Projection of generalized rotational states comes from the product of
    // Stiefel manifolds
    P.block(size_, 0, n_[0], r_) =
        OB_.Proj(Y.block(size_, 0, n_[0], r_), dotY.block(size_, 0, n_[0], r_));

    return P;
  }
}

ComplexMatrix CPL_SLAMProblem::retract(const ComplexMatrix &Y,
                                      const ComplexMatrix &dotY) const {
  if (form_ == Formulation::Simplified)
    return (OB_.*retract_)(Y, dotY);
  else  // form == Explicit
  {
    ComplexMatrix Yplus = Y;
    Yplus.block(0, 0, size_, r_) += dotY.block(0, 0, size_, r_);

    Yplus.block(size_, 0, n_[0], r_) = (OB_.*retract_)(
        Y.block(size_, 0, n_[0], r_), dotY.block(size_, 0, n_[0], r_));

    return Yplus;
  }
}

ComplexVector CPL_SLAMProblem::round_solution(const ComplexMatrix Y) const {
  // First, compute a thin SVD of Y
  Eigen::JacobiSVD<ComplexMatrix> svd(Y, Eigen::ComputeThinU);

  Scalar sigma = svd.singularValues()[0];

  // First, construct a rank-d truncated singular value decomposition for Y
  ComplexVector R = sigma * svd.matrixU().leftCols(d_);

  R.tail(n_[0]).noalias() = R.tail(n_[0]).rowwise().normalized();

  if (form_ == Formulation::Explicit)
    return R;
  else  // form == Explicit
  {
    // In this case, we also need to recover the corresponding translations
    ComplexVector X(2 * n_[0] + n_[1], 1);

    // Set rotational states
    X.tail(n_[0]) = R;

    // Recover translational states
    X.head(size_) = recover_landmarks_and_translations(B1_, B2_, R);

    return X;
  }
}

RealDiagonalMatrix CPL_SLAMProblem::compute_Lambda(
    const ComplexMatrix &Y) const {
  // First, compute the diagonal blocks of Lambda
  ComplexMatrix SY = data_matrix_product(Y);
  RealDiagonalMatrix Lambda(n_[0]);

  RealDiagonalMatrix::DiagonalVectorType &diagonal = Lambda.diagonal();

  const size_t offset = form_ == Formulation::Simplified ? 0 : n_[0] + n_[1];

#pragma omp parallel for
  for (size_t i = 0; i < n_[0]; ++i) {
    diagonal[i] = SY.row(offset + i).dot(Y.row(offset + i)).real();
  }

  return Lambda;
}

bool CPL_SLAMProblem::compute_S_minus_Lambda_min_eig(
    const ComplexMatrix &Y, Scalar &min_eigenvalue,
    ComplexVector &min_eigenvector, size_t &num_iterations,
    size_t max_iterations, Scalar min_eigenvalue_nonnegativity_tolerance,
    size_t num_Lanczos_vectors) const {
  // First, compute the largest-magnitude eigenvalue of this matrix
  SMinusLambdaProdFunctor lm_op(this, Y);
  Spectra::SymEigsSolver<Scalar, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN,
                         SMinusLambdaProdFunctor>
      largest_magnitude_eigensolver(&lm_op, 1,
                                    std::min(num_Lanczos_vectors, n_[0]));
  largest_magnitude_eigensolver.init();

  int num_converged = largest_magnitude_eigensolver.compute(
      max_iterations, 1e-4, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN);

  // Check convergence and bail out if necessary
  if (num_converged != 1) return false;

  const size_t size =
      form_ == Formulation::Simplified ? n_[0] : 2 * n_[0] + n_[1];

  min_eigenvector.resize(size);
  Eigen::Map<RealVector> real_min_eigenvector((Scalar *)min_eigenvector.data(),
                                              2 * size);

  Scalar lambda_lm = largest_magnitude_eigensolver.eigenvalues()[0];

  if (lambda_lm < 0) {
    // The largest-magnitude eigenvalue is negative, and therefore also the
    // minimum eigenvalue, so just return this solution
    min_eigenvalue = lambda_lm;
    real_min_eigenvector = largest_magnitude_eigensolver.eigenvectors(1);
    min_eigenvector.normalize();  // Ensure that this is a unit vector
    return true;
  }

  // The largest-magnitude eigenvalue is positive, and is therefore the
  // maximum  eigenvalue.  Therefore, after shifting the spectrum of S - Lambda
  // by -2*lambda_lm (by forming S - Lambda - 2*lambda_max*I), the  shifted
  // spectrum will lie in the interval [lambda_min(A) - 2*  lambda_max(A),
  // -lambda_max*A]; in particular, the largest-magnitude eigenvalue of  S -
  // Lambda - 2*lambda_max*I is lambda_min - 2*lambda_max, with  corresponding
  // eigenvector v_min; furthermore, the condition number sigma of S - Lambda
  // -2*lambda_max is then upper-bounded by 2 :-).

  SMinusLambdaProdFunctor min_shifted_op(this, Y, -2 * lambda_lm);

  Spectra::SymEigsSolver<Scalar, Spectra::SELECT_EIGENVALUE::LARGEST_MAGN,
                         SMinusLambdaProdFunctor>
      min_eigensolver(&min_shifted_op, 1, std::min(num_Lanczos_vectors, n_[0]));

  // If Y is a critical point of F, then Y^T is also in the null space of S -
  // Lambda(Y) (cf. Lemma 6 of the tech report), and therefore its rows are
  // eigenvectors corresponding to the eigenvalue 0.  In the case  that the
  // relaxation is exact, this is the *minimum* eigenvalue, and therefore the
  // rows of Y are exactly the eigenvectors that we're looking for.  On the
  // other hand, if the relaxation is *not* exact, then S - Lambda(Y) has at
  // least one strictly negative eigenvalue, and the rows of Y are *unstable
  // fixed points* for the Lanczos iterations.  Thus, we will take a slightly
  // "fuzzed" version of the first row of Y as an initialization for the Lanczos
  // iterations; this allows for rapid convergence in the case that the
  // relaxation is exact (since are starting close to a solution), while
  // simultaneously allowing the iterations to escape from this fixed point in
  // the case that the relaxation is not exact.
  ComplexVector v0 = Y.col(0);

  Scalar d = (form_ == Formulation::Simplified) ? .01 : .03;

  v0.noalias() +=
      (d * v0.norm()) * ComplexVector::Random(v0.size()).normalized();

  // Use this to initialize the eigensolver
  min_eigensolver.init((Scalar *)v0.data());

  // Now determine the relative precision required in the Lanczos method in
  // order to be able to estimate the smallest eigenvalue within an *absolute*
  // tolerance of 'min_eigenvalue_nonnegativity_tolerance'
  num_converged = min_eigensolver.compute(
      max_iterations, min_eigenvalue_nonnegativity_tolerance / lambda_lm,
      Spectra::SELECT_EIGENVALUE::LARGEST_MAGN);

  if (num_converged != 1) return false;

  real_min_eigenvector = min_eigensolver.eigenvectors(1);
  min_eigenvector.normalize();
  min_eigenvalue = min_eigensolver.eigenvalues()(0) + 2 * lambda_lm;
  num_iterations = min_eigensolver.num_iterations();

  return true;
}

ComplexMatrix CPL_SLAMProblem::chordal_initialization() const {
  ComplexMatrix Y;
  if (form_ == Formulation::Simplified) {
    Y = ComplexMatrix::Zero(n_[0], r_);

    if (B1_.cols() == B3_.cols())
      Y.leftCols(d_) = CPL_SLAM::chordal_initialization(B3_);
    else
      Y.leftCols(d_) = CPL_SLAM::chordal_initialization(B1_, B2_, B3_);
  } else  // form == explicit
  {
    Y = ComplexMatrix::Zero(2 * n_[0] + n_[1], r_);

    // Compute rotations using chordal initialization
    if (B1_.cols() == B3_.cols())
      Y.block(size_, 0, n_[0], d_) = CPL_SLAM::chordal_initialization(B3_);
    else
      Y.block(size_, 0, n_[0], d_) =
          CPL_SLAM::chordal_initialization(B1_, B2_, B3_);

    // Recover corresponding translations
    Y.block(0, 0, size_, d_) = recover_landmarks_and_translations(
        B1_, B2_, Y.block(size_, 0, n_[0], d_));
  }

  return Y;
}

ComplexMatrix CPL_SLAMProblem::random_sample() const {
  ComplexMatrix Y;
  if (form_ == Formulation::Simplified)
    // Randomly sample a point on the Stiefel manifold
    Y = OB_.random_sample();
  else  // form == Explicit
  {
    Y = ComplexMatrix::Zero(r_, n_[0] * (d_ + 1) + n_[1]);

    // Randomly sample a set of elements on the Stiefel product manifold
    Y.block(size_, 0, n_[0], r_) = OB_.random_sample();

    // Randomly sample a set of coordinates for the initial positions from the
    // standard normal distribution
    std::default_random_engine generator;
    std::normal_distribution<Scalar> g;

    for (size_t i = 0; i < r_; ++i)
      for (size_t j = 0; j < size_; ++j)
        Y(i, j) = Complex(g(generator), g(generator));
  }

  return Y;
}

/// COMPLEX MATRIX EIGENVALUE COMPUTATION

CPL_SLAMProblem::SparseClpMatProd::SparseClpMatProd(
    const ComplexSparseMatrix &mat)
    : mat_(mat) {
  rows_ = 2 * mat.rows();
  cols_ = 2 * mat.cols();
}

void CPL_SLAMProblem::SparseClpMatProd::perform_op(const Scalar *x,
                                                  Scalar *y) const {
  Eigen::Map<const ComplexVector> X((Complex *)x, mat_.cols());
  Eigen::Map<ComplexVector> Y((Complex *)y, mat_.rows());

  Y.noalias() = mat_ * X;
}

/// MINIMUM EIGENVALUE COMPUTATION STRUCT

CPL_SLAMProblem::SMinusLambdaProdFunctor::SMinusLambdaProdFunctor(
    const CPL_SLAMProblem *prob, const ComplexMatrix &Y, Scalar sigma)
    : problem_(prob), dim_(prob->dimension()), sigma_(sigma) {
  if (problem_->formulation() == Formulation::Simplified) {
    n_ = problem_->dimension() * problem_->num_poses();
  } else  // mode == Explicit
  {
    n_ = problem_->dimension() *
         (2 * problem_->num_poses() + problem_->num_landmarks());
  }

  rows_ = 2 * n_;
  cols_ = 2 * n_;

  // Compute and cache this on construction
  Lambda_ = problem_->compute_Lambda(Y);
}

void CPL_SLAMProblem::SMinusLambdaProdFunctor::perform_op(const Scalar *x,
                                                         Scalar *y) const {
  Eigen::Map<const ComplexVector> X((Complex *)x, n_);
  Eigen::Map<ComplexVector> Y((Complex *)y, n_);

  Y = problem_->data_matrix_product(X);

  size_t const &nn = problem_->num_poses();

  Y.tail(nn).noalias() -= Lambda_ * X.tail(nn);

  if (sigma_ != 0) Y += sigma_ * X;
}
}  // namespace CPL_SLAM
