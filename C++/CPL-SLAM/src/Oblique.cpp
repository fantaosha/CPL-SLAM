
#include <Eigen/QR>
#include <Eigen/SVD>

#include "CPL-SLAM/Oblique.h"
namespace CPL_SLAM {

ComplexMatrix Oblique::project(const ComplexMatrix &A) const {

// We use a generalization of the well-known SVD-based projection for the
// orthogonal and special orthogonal groups; see for example Proposition 7
// in the paper "Projection-Like Retractions on ComplexMatrix Manifolds" by
// Absil
// and Malick.

#if defined(_OPENMP)
  ComplexMatrix P(n_, p_);

#pragma omp parallel for
  for (size_t i = 0; i < n_; ++i) {
    P.row(i).noalias() = A.row(i).normalized();
  }
  return P;
#else
  return A.rowwise().normalized();
#endif
}

ComplexMatrix Oblique::DDiagProduct(const ComplexMatrix &A,
                                    const ComplexMatrix &B,
                                    const ComplexMatrix &C) const {
  ComplexMatrix D(n_, p_);

#pragma omp parallel for
  for (size_t i = 0; i < n_; ++i) {
    D.row(i) = A.row(i).dot(B.row(i)).real() * C.row(i);
  }

  return D;
}

ComplexMatrix Oblique::Proj(const ComplexMatrix &Y,
                            const ComplexMatrix &V) const {
  return V - DDiagProduct(Y, V, Y);
}

ComplexMatrix Oblique::retract(const ComplexMatrix &Y,
                               const ComplexMatrix &V) const {

  // We use projection-based retraction, as described in "Projection-Like
  // Retractions on ComplexMatrix Manifolds" by Absil and Malick

  return project(Y + V);
}

ComplexMatrix Oblique::exp(const ComplexMatrix &Y,
                           const ComplexMatrix &V) const {
  // Preallocate result matrix
  ComplexMatrix P(n_, p_);

  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> theta = V.rowwise().norm();
#pragma omp parallel for
  for (size_t i = 0; i < n_; ++i) {

    const Scalar &t = theta[i];
    if (t < std::numeric_limits<Scalar>::epsilon()) {

      double t2 = t * t;
      double t4 = t2 * t2;

      double c = 1 - t2 / 2 + t4 / 24;
      double s = 1 - t2 / 6 + t4 / 120;

      P.row(i).noalias() = Y.row(i) * c + V.row(i) * s;
    } else {
      double c = cos(t);
      double s = sin(t) / t;

      P.row(i).noalias() = Y.row(i) * c + V.row(i) * s;
    }
  }

  return P;
}

ComplexMatrix Oblique::random_sample(
    const std::default_random_engine::result_type &seed) const {
  // Generate a matrix of the appropriate dimension by sampling its elements
  // from the standard Gaussian
  std::default_random_engine generator(seed);
  std::normal_distribution<Scalar> g;

  ComplexMatrix R(p_, n_);
  for (size_t r = 0; r < p_; ++r)
    for (size_t c = 0; c < n_; ++c)
      R(r, c) = Complex(g(generator), g(generator));
  return project(R);
}
} // namespace CPL_SLAM
