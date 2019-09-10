/** This lightweight class models the geometry of complex oblique
 * manifold.
 *
 * Copyright (C) 2018 - 2019 by Taosha Fan (taosha.fan@gmail.com)
 */

#pragma once

#include <random> // For sampling random points on the manifold

#include <Eigen/Dense>

#include "CPL-SLAM/CPL-SLAM_types.h"

namespace CPL_SLAM {

class Oblique {

private:
  // Dimension of ambient Euclidean space containing the frames
  size_t p_;

  // Number of copies of St(k,p) in the product
  size_t n_;

public:
  /// CONSTRUCTORS AND MUTATORS

  // Default constructor -- sets all dimensions to 0
  Oblique() {}

  Oblique(size_t p, size_t n) : p_(p), n_(n) {}

  void set_p(size_t p) { p_ = p; }
  void set_n(size_t n) { n_ = n; }

  /// ACCESSORS
  unsigned int get_p() const { return p_; }
  unsigned int get_n() const { return n_; }

  /// GEOMETRY

  /** Given a generic matrix A in R^{p x kn}, this function computes the
   * projection of A onto R (closest point in the Frobenius norm sense).  */
  ComplexMatrix project(const ComplexMatrix &A) const;

  /** Helper function -- this computes and returns the product
   *
   *  P = real(DDiag(A * B^H)) * C
   *
   * where A, B, and C are p x kn matrices (cf. eq. (5) in the CPL-SLAM tech
   * report).
   */
  ComplexMatrix DDiagProduct(const ComplexMatrix &A, const ComplexMatrix &B,
                             const ComplexMatrix &C) const;

  /** Given an element Y in M and a matrix V in T_X(R^{p x kn}) (that is, a (p
   * x kn)-dimensional matrix V considered as an element of the tangent space to
   * the *entire* ambient Euclidean space at X), this function computes and
   * returns the projection of V onto T_X(M), the tangent space of M at X (cf.
   * eq. (42) in the CPL-SLAM tech report).*/
  ComplexMatrix Proj(const ComplexMatrix &Y, const ComplexMatrix &V) const;

  /** Given an element Y in M and a tangent vector V in T_Y(M), this function
   * computes the retraction along V at Y using the QR-based retraction
   * specified in eq. (4.8) of Absil et al.'s  "Optimization Algorithms on
   * ComplexMatrix Manifolds").
   */
  ComplexMatrix retract(const ComplexMatrix &Y, const ComplexMatrix &V) const;

  ComplexMatrix exp(const ComplexMatrix &Y, const ComplexMatrix &V) const;

  /** Sample a random point on M, using the (optional) passed seed to initialize
   * the random number generator.  */
  ComplexMatrix
  random_sample(const std::default_random_engine::result_type &seed =
                    std::default_random_engine::default_seed) const;
};

} // namespace CPL_SLAM
