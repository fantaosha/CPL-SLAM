#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

#include "CPL-SLAM/CPL-SLAM_utils.h"

namespace CPL_SLAM {

measurements_t read_g2o_file(const std::string &filename) {
  measurements_t measurements;

  // Preallocate output
  measurements_pose_t &measurements_pose = measurements.first;
  measurements_landmark_t &measurements_landmark = measurements.second;
  size_t &num_poses = measurements.num_poses;
  size_t &num_landmarks = measurements.num_landmarks;

  CPL_SLAM::RelativePoseMeasurement measurement_pose;
  CPL_SLAM::RelativeLandmarkMeasurement measurement_landmark;

  // A single measurement, whose values we will fill in

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

  Scalar ss, cc;

  size_t i, j;

  // Open the file for reading
  std::ifstream infile(filename);

  num_poses = 0;
  num_landmarks = 0;

  std::unordered_map<size_t, size_t> poses;
  std::unordered_map<size_t, size_t> landmarks;

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::stringstream strstrm(line);

    // Extract the first token from the string
    strstrm >> token;

    if (token == "EDGE_SE2") {
      // This is a 2D pose measurement

      /** The g2o format specifies a 2D relative pose measurement in the
       * following form:
       *
       * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
       *
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
          I23 >> I33;

      if (poses.insert({i, num_poses}).second) num_poses++;

      if (poses.insert({j, num_poses}).second) num_poses++;

      // Fill in elements of this measurement

      // Pose ids
      measurement_pose.i = poses[i];
      measurement_pose.j = poses[j];

      // Raw measurements
      sincos(dtheta, &ss, &cc);
      measurement_pose.t = Complex(dx, dy);
      measurement_pose.R = Complex(cc, ss);

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement_pose.tau = 2 / TranCov.inverse().trace();

      measurement_pose.kappa = 2 * I33;

      // Update maximum value of poses found so far
      measurements_pose.push_back(measurement_pose);

    } else if (token == "VERTEX_SE2") {
      continue;
    } else if (token == "POINT2") {
      continue;
    } else if (token == "LANDMARK2") {
      strstrm >> i >> j >> dx >> dy >> I11 >> I12 >> I22;

      if (poses.insert({i, num_poses}).second) num_poses++;

      if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;

      measurement_landmark.i = poses[i];
      measurement_landmark.j = landmarks[j];

      measurement_landmark.l = Complex(dx, dy);

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement_landmark.nu = 2 / TranCov.inverse().trace();

      measurements_landmark.push_back(measurement_landmark);
    }  else {
      std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
      assert(false);
    }
  }  // while

  infile.close();

  return measurements;
}

ComplexSparseMatrix construct_rotational_connection_Laplacian(
    const measurements_t &measurements) {
  const size_t &num_poses = measurements.num_poses;
  // Each measurement contributes 2*d elements along the diagonal of the
  // connection Laplacian, and 2*d^2 elements on a pair of symmetric
  // off-diagonal blocks

  size_t measurement_stride = 4;

  std::vector<Eigen::Triplet<Complex>> triplets;
  triplets.reserve(measurement_stride * measurements.first.size());

  size_t i, j;

  for (const CPL_SLAM::RelativePoseMeasurement &measurement :
       measurements.first) {
    i = measurement.i;
    j = measurement.j;

    // Elements of ith block-diagonal
    triplets.emplace_back(i, i, measurement.kappa);

    // Elements of jth block-diagonal
    triplets.emplace_back(j, j, measurement.kappa);

    // Elements of ij block
    triplets.emplace_back(i, j, -measurement.kappa * std::conj(measurement.R));

    // Elements of ji block
    triplets.emplace_back(j, i, -measurement.kappa * measurement.R);
  }

  // Construct and return a sparse matrix from these triplets
  ComplexSparseMatrix LGz(num_poses, num_poses);
  LGz.setFromTriplets(triplets.begin(), triplets.end());

  return LGz;
}

RealSparseMatrix construct_oriented_incidence_matrix(
    const measurements_t &measurements) {
  std::vector<Eigen::Triplet<Scalar>> triplets;

  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  triplets.reserve(2 * num_measurements);
  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    triplets.emplace_back(measurements.second[m].i + num_landmarks, e, -1);
    triplets.emplace_back(measurements.second[m].j, e, 1);
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    triplets.emplace_back(measurements.first[m].i + num_landmarks, e, -1);
    triplets.emplace_back(measurements.first[m].j + num_landmarks, e, 1);
  }

  RealSparseMatrix A(num_poses + num_landmarks, num_measurements);
  A.setFromTriplets(triplets.begin(), triplets.end());

  return A;
}

RealDiagonalMatrix construct_landmark_and_translational_precision_matrix(
    const measurements_t &measurements) {
  // Allocate output matrix
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  RealDiagonalMatrix Omega(num_measurements);

  RealDiagonalMatrix::DiagonalVectorType &diagonal = Omega.diagonal();

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++)
    diagonal[e] = measurements.second[m].nu;

  for (size_t m = 0; e < num_measurements; e++, m++)
    diagonal[e] = measurements.first[m].tau;

  return Omega;
}

ComplexSparseMatrix construct_landmark_and_translational_data_matrix(
    const measurements_t &measurements) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_landmarks = measurements.num_landmarks;

  std::vector<Eigen::Triplet<Complex>> triplets;
  triplets.reserve(num_measurements);

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    triplets.emplace_back(e, measurements.second[m].i,
                          -measurements.second[m].l);
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    triplets.emplace_back(e, measurements.first[m].i, -measurements.first[m].t);
  }

  ComplexSparseMatrix T(num_measurements, measurements.num_poses);
  T.setFromTriplets(triplets.begin(), triplets.end());

  return T;
}

void construct_B_matrices(const measurements_t &measurements,
                          ComplexSparseMatrix &B1, ComplexSparseMatrix &B2,
                          ComplexSparseMatrix &B3) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  // Clear input matrices
  B1.setZero();
  B2.setZero();
  B3.setZero();

  std::vector<Eigen::Triplet<Complex>> triplets;

  // Useful quantities to cache

  size_t i, j;  // Indices for the tail and head of the given measurement
  Scalar sqrtnu, sqrttau;

  /// Construct the matrix B1 from equation (69a) in the tech report
  triplets.reserve(2 * num_measurements);

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    i = measurements.second[m].i;
    j = measurements.second[m].j;
    sqrtnu = sqrt(measurements.second[m].nu);

    // Block corresponding to the tail of the measurement
    triplets.emplace_back(e, i + num_landmarks,
                          -sqrtnu);  // Diagonal element corresponding to tail
    triplets.emplace_back(e, j,
                          sqrtnu);  // Diagonal element corresponding to head
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    i = measurements.first[m].i;
    j = measurements.first[m].j;
    sqrttau = sqrt(measurements.first[m].tau);

    // Block corresponding to the tail of the measurement
    triplets.emplace_back(e, i + num_landmarks,
                          -sqrttau);  // Diagonal element corresponding to tail
    triplets.emplace_back(e, j + num_landmarks,
                          sqrttau);  // Diagonal element corresponding to head
  }

  B1.resize(num_measurements, num_poses + num_landmarks);
  B1.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B2 from equation (69b) in the tech report
  triplets.clear();
  triplets.reserve(num_measurements);

  e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    i = measurements.second[m].i;
    sqrtnu = sqrt(measurements.second[m].nu);
    triplets.emplace_back(e, i, -sqrtnu * measurements.second[m].l);
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    i = measurements.first[m].i;
    sqrttau = sqrt(measurements.first[m].tau);
    triplets.emplace_back(e, i, -sqrttau * measurements.first[m].t);
  }

  B2.resize(num_measurements, num_poses);
  B2.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B3 from equation (69c) in the tech report
  triplets.clear();
  triplets.reserve(2 * measurements.first.size());

  e = 0;

  for (size_t m = 0; e < measurements.first.size(); e++, m++) {
    Scalar sqrtkappa = std::sqrt(measurements.first[m].kappa);
    auto const &R = measurements.first[m].R;

    i = measurements.first[m].i;  // Tail of measurement
    j = measurements.first[m].j;  // Head of measurement

    // Representation of the -sqrt(kappa) * Rt(i,j) \otimes I_d block
    triplets.emplace_back(e, i, -sqrtkappa * R);

    triplets.emplace_back(e, j, sqrtkappa);
  }

  B3.resize(measurements.first.size(), num_poses);
  B3.setFromTriplets(triplets.begin(), triplets.end());
}

ComplexSparseMatrix construct_quadratic_form_data_matrix(
    const measurements_t &measurements) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  std::vector<Eigen::Triplet<Complex>> triplets;

  // Number of nonzero elements contributed to Sigma^s by each measurement
  size_t Sigmas_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to U by each measurement
  size_t U_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to N by each measurement
  size_t N_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to L(W^c) by each measurement
  size_t LWc_nnz_per_measurement[2] = {4, 0};

  // Number of nonzero elements contributed to Sigma^s by each measurement
  size_t Sigmac_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to V by each measurement
  size_t E_nnz_per_measurement[2] = {2, 1};

  // Number of nonzero elements contributed to L(G^z) by each measurement
  size_t LGz_nnz_per_measurement[2] = {4, 0};

  // Number of nonzero elements contributed to Sigma^z by each measurement
  size_t Sigmaz_nnz_per_measurement[2] = {1, 1};

  // Number of nonzero elements contributed to the entire matrix M by each
  // measurement
  size_t num_nnz_per_measurement[2] = {
      Sigmas_nnz_per_measurement[0] + 2 * U_nnz_per_measurement[0] +
          2 * N_nnz_per_measurement[0] + LWc_nnz_per_measurement[0] +
          Sigmac_nnz_per_measurement[0] + 2 * E_nnz_per_measurement[0] +
          LGz_nnz_per_measurement[0] + Sigmaz_nnz_per_measurement[0],
      Sigmas_nnz_per_measurement[1] + 2 * U_nnz_per_measurement[1] +
          2 * N_nnz_per_measurement[1] + LWc_nnz_per_measurement[1] +
          Sigmac_nnz_per_measurement[1] + 2 * E_nnz_per_measurement[1] +
          LGz_nnz_per_measurement[1] + Sigmaz_nnz_per_measurement[1]};

  /// Working space
  size_t i, j;  // Indices for the tail and head of the given measurement

  triplets.reserve(num_nnz_per_measurement[0] * measurements.first.size() +
                   num_nnz_per_measurement[1] * measurements.second.size());

  // Now scan through the measurements again, using knowledge of the total
  // number of poses to compute offsets as appropriate

  for (const CPL_SLAM::RelativePoseMeasurement &measurement :
       measurements.first) {
    i = measurement.i + num_landmarks;  // Tail of measurement
    j = measurement.j + num_landmarks;  // Head of measurement

    // Add elements for L(W^tau)
    triplets.emplace_back(i, i, measurement.tau);
    triplets.emplace_back(j, j, measurement.tau);
    triplets.emplace_back(j, i, -measurement.tau);
    triplets.emplace_back(i, j, -measurement.tau);

    // Add elements for E (upper-right block)
    triplets.emplace_back(i, num_poses + i, measurement.tau * measurement.t);
    triplets.emplace_back(j, num_poses + i, -measurement.tau * measurement.t);

    // Add elements for E' (lower-left block)
    triplets.emplace_back(num_poses + i, i,
                          measurement.tau * std::conj(measurement.t));
    triplets.emplace_back(num_poses + i, j,
                          -measurement.tau * std::conj(measurement.t));

    // Add elements for L(G^rho)
    // Elements of ith block-diagonal
    triplets.emplace_back(num_poses + i, num_poses + i, measurement.kappa);

    // Elements of jth block-diagonal
    triplets.emplace_back(num_poses + j, num_poses + j, measurement.kappa);

    // Elements of ij block
    triplets.emplace_back(num_poses + i, num_poses + j,
                          -measurement.kappa * std::conj(measurement.R));

    // Elements of ji block
    triplets.emplace_back(num_poses + j, num_poses + i,
                          -measurement.kappa * measurement.R);

    // Add elements for Sigma
    triplets.emplace_back(num_poses + i, num_poses + i,
                          measurement.tau * std::norm(measurement.t));
  }

  for (const CPL_SLAM::RelativeLandmarkMeasurement &measurement :
       measurements.second) {
    i = measurement.i + num_landmarks;  // Tail of measurement
    j = measurement.j;                  // Head of measurement

    // Add elements for Sigma^s
    triplets.emplace_back(j, j, measurement.nu);

    // Add elements for U
    triplets.emplace_back(j, i, -measurement.nu);

    // Add elements for U'
    triplets.emplace_back(i, j, -measurement.nu);

    // Add elements for N
    triplets.emplace_back(j, i + num_poses, -measurement.nu * measurement.l);

    // Add elements for N'
    triplets.emplace_back(i + num_poses, j,
                          -measurement.nu * std::conj(measurement.l));

    // Add elements for Sigma^c
    triplets.emplace_back(i, i, measurement.nu);

    // Add elements for E
    triplets.emplace_back(i, i + num_poses, measurement.nu * measurement.l);

    // Add elements for E'
    triplets.emplace_back(i + num_poses, i,
                          measurement.nu * std::conj(measurement.l));

    // Add elements for Sigma^z
    triplets.emplace_back(i + num_poses, i + num_poses,
                          measurement.nu * std::norm(measurement.l));
  }

  ComplexSparseMatrix M(num_landmarks + 2 * num_poses,
                        num_landmarks + 2 * num_poses);
  M.setFromTriplets(triplets.begin(), triplets.end());

  return M;
}

ComplexVector chordal_initialization(const ComplexSparseMatrix &B3) {
  size_t num_poses = B3.cols();

  /// We want to find a minimizer of
  /// || B3 * r ||
  ///
  /// For the purposes of initialization, we can simply fix the first pose to
  /// the origin; this corresponds to fixing the first d^2 elements of r to
  /// vec(I_d), and slicing off the first d^2 columns of B3 to form
  ///
  /// min || B3red * rred + c ||, where
  ///
  /// c = B3(1:d^2) * vec(I_3)

  ComplexSparseMatrix B3red = B3.rightCols((num_poses - 1));
  // Must be in compressed format to use Eigen::SparseQR!
  B3red.makeCompressed();

  ComplexVector cR = B3.leftCols(1);

  Eigen::SPQR<ComplexSparseMatrix> QR(B3red);

  ComplexVector Rchordal(num_poses, 1);
  Rchordal(0) = 1;
  Rchordal.tail(num_poses - 1) = -QR.solve(cR);

  // Rchordal.array() /= Rchordal.array().abs();
  Rchordal.tail(num_poses - 1).rowwise().normalize();

  return Rchordal;
}

ComplexVector chordal_initialization(const ComplexSparseMatrix &B1,
                                     const ComplexSparseMatrix &B2,
                                     const ComplexSparseMatrix &B3) {
  const size_t num_poses = B3.cols();
  const size_t size = B1.cols();

  CPL_SLAM::ComplexVector b(B1.rows() + B3.rows());
  b.head(B2.rows()) = -B2.col(0);
  b.tail(B3.rows()) = -B3.col(0);

  CPL_SLAM::ComplexSparseMatrix Bred(B1.rows() + B3.rows(),
                                    size + num_poses - 2);

  std::vector<Eigen::Triplet<CPL_SLAM::Complex>> triplets;
  triplets.reserve(B1.nonZeros() + B2.nonZeros() + B3.nonZeros());

  for (int k = 0; k < B1.outerSize(); k++) {
    for (ComplexSparseMatrix::InnerIterator it(B1, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j == 0) continue;

      triplets.emplace_back(i, j - 1, value);
    }
  }

  for (int k = 0; k < B2.outerSize(); k++) {
    for (ComplexSparseMatrix::InnerIterator it(B2, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j == 0) continue;

      triplets.emplace_back(i, j + size - 2, value);
    }
  }

  for (int k = 0; k < B3.outerSize(); k++) {
    for (ComplexSparseMatrix::InnerIterator it(B3, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j == 0) continue;

      triplets.emplace_back(i + B1.rows(), j + size - 2, value);
    }
  }

  Bred.setFromTriplets(triplets.begin(), triplets.end());

  ComplexVector Rchordal(num_poses, 1);
  Rchordal(0) = 1;

  Eigen::SPQR<ComplexSparseMatrix> QR(Bred);
  Rchordal.tail(num_poses - 1) =
      QR.solve(b).tail(num_poses - 1).rowwise().normalized();

  return Rchordal;
}

ComplexVector recover_landmarks_and_translations(const ComplexSparseMatrix &B1,
                                                 const ComplexSparseMatrix &B2,
                                                 const ComplexVector &R) {
  size_t n = B1.cols();

  /// We want to find a minimizer of
  /// || B1 * t + B2 * vec(R) ||
  ///
  /// For the purposes of initialization, we can simply fix the first pose to
  /// the origin; this corresponds to fixing the first d elements of t to 0,
  /// and
  /// slicing off the first d columns of B1 to form
  ///
  /// min || B1red * tred + c) ||, where
  ///
  /// c = B2 * vec(R)

  // Form the matrix comprised of the right (n-1) block columns of B1
  ComplexSparseMatrix B1red = B1.rightCols(n - 1);

  ComplexVector c = B2 * R;

  // Solve
  Eigen::SPQR<ComplexSparseMatrix> QR(B1red);
  ComplexVector tred = -QR.solve(c);

  // Allocate output matrix
  ComplexMatrix t = ComplexMatrix::Zero(n, 1);

  // Set rightmost n-1 columns
  t.bottomRows(n - 1) = tred;

  return t;
}
}  // namespace CPL_SLAM
