#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>

#include "SESync/SESync_utils.h"

namespace SESync {

measurements_t read_g2o_file(const std::string &filename) {
  // Preallocate output vector
  measurements_t measurements;

  measurements_pose_t &measurements_pose = measurements.first;
  measurements_landmark_t &measurements_landmark = measurements.second;
  size_t &num_poses = measurements.num_poses;
  size_t &num_landmarks = measurements.num_landmarks;

  // A single measurement, whose values we will fill in
  SESync::RelativePoseMeasurement measurement_pose;
  SESync::RelativeLandmarkMeasurement measurement_landmark;

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  Scalar dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;

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
      measurements.dimension = 2;
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
      measurement_pose.t = Eigen::Matrix<Scalar, 2, 1>(dx, dy);
      measurement_pose.R = Eigen::Rotation2D<Scalar>(dtheta).toRotationMatrix();

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement_pose.tau = 2 / TranCov.inverse().trace();

      measurement_pose.kappa = I33;

      measurements_pose.push_back(measurement_pose);

    } else if (token == "EDGE_SE3:QUAT") {
      measurements.dimension = 3;
      // This is a 3D pose measurement

      /** The g2o format specifies a 3D relative pose measurement in the
       * following form:
       *
       * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
       *
       * I11 I12 I13 I14 I15 I16
       *     I22 I23 I24 I25 I26
       *         I33 I34 I35 I36
       *             I44 I45 I46
       *                 I55 I56
       *                     I66
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
          I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
          I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

      if (poses.insert({i, num_poses}).second) num_poses++;

      if (poses.insert({j, num_poses}).second) num_poses++;

      // Fill in elements of this measurement

      // Pose ids
      measurement_pose.i = poses[i];
      measurement_pose.j = poses[j];

      // Raw measurements
      measurement_pose.t = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);
      measurement_pose.R =
          Eigen::Quaternion<Scalar>(dqw, dqx, dqy, dqz).toRotationMatrix();

      // Compute precisions

      // Compute and store the optimal (information-divergence-minimizing) value
      // of the parameter tau
      Eigen::Matrix<Scalar, 3, 3> TranCov;
      TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;
      measurement_pose.tau = 3 / TranCov.inverse().trace();

      // Compute and store the optimal (information-divergence-minimizing value
      // of the parameter kappa

      Eigen::Matrix<Scalar, 3, 3> RotCov;
      RotCov << I44, I45, I46, I45, I55, I56, I46, I56, I66;
      measurement_pose.kappa = 3 / (2 * RotCov.inverse().trace());

      measurements_pose.push_back(measurement_pose);

    } else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT") ||
               (token == "POINT2")) {
      // This is just initialization information, so do nothing
      continue;
    } else if (token == "LANDMARK2") {
      strstrm >> i >> j >> dx >> dy >> I11 >> I12 >> I22;

      if (poses.insert({i, num_poses}).second) num_poses++;

      if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;

      measurement_landmark.i = poses[i];
      measurement_landmark.j = landmarks[j];

      measurement_landmark.l = Eigen::Matrix<Scalar, 2, 1>(dx, dy);

      Eigen::Matrix<Scalar, 2, 2> TranCov;
      TranCov << I11, I12, I12, I22;
      measurement_landmark.nu = 2 / TranCov.inverse().trace();

      measurements_landmark.push_back(measurement_landmark);
    } else if (token == "LANDMARK3") {
      strstrm >> i >> j >> dx >> dy >> dz >> I11 >> I12 >> I13 >> I22 >> I23 >>
          I33;

      if (poses.insert({i, num_poses}).second) num_poses++;

      if (landmarks.insert({j, num_landmarks}).second) num_landmarks++;

      measurement_landmark.i = poses[i];
      measurement_landmark.j = landmarks[j];

      measurement_landmark.l = Eigen::Matrix<Scalar, 3, 1>(dx, dy, dz);

      Eigen::Matrix<Scalar, 3, 3> TranCov;
      TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;

      measurement_landmark.nu = 3 / (2 * TranCov.inverse().trace());
    } else {
      std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
      assert(false);
    }

  }  // while

  infile.close();

  return measurements;
}

SparseMatrix construct_rotational_connection_Laplacian(
    const measurements_t &measurements) {
  const size_t &num_poses =
      measurements
          .num_poses;  // We will use this to keep track of the largest pose
  // index encountered, which in turn provides the number
  // of poses

  size_t d = (!measurements.first.empty() ? measurements.first[0].t.size() : 0);

  // Each measurement contributes 2*d elements along the diagonal of the
  // connection Laplacian, and 2*d^2 elements on a pair of symmetric
  // off-diagonal blocks

  size_t measurement_stride = 2 * (d + d * d);

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(measurement_stride * measurements.first.size());

  size_t i, j;
  for (const SESync::RelativePoseMeasurement &measurement :
       measurements.first) {
    i = measurement.i;
    j = measurement.j;

    // Elements of ith block-diagonal
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(d * i + k, d * i + k, measurement.kappa);

    // Elements of jth block-diagonal
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(d * j + k, d * j + k, measurement.kappa);

    // Elements of ij block
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(i * d + r, j * d + c,
                              -measurement.kappa * measurement.R(r, c));

    // Elements of ji block
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(j * d + r, i * d + c,
                              -measurement.kappa * measurement.R(c, r));
  }

  // Construct and return a sparse matrix from these triplets
  SparseMatrix LGrho(d * num_poses, d * num_poses);
  LGrho.setFromTriplets(triplets.begin(), triplets.end());

  return LGrho;
}

SparseMatrix construct_oriented_incidence_matrix(
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

  SparseMatrix A(num_poses + num_landmarks, num_measurements);
  A.setFromTriplets(triplets.begin(), triplets.end());
  return A;
}

DiagonalMatrix construct_landmark_and_translational_precision_matrix(
    const measurements_t &measurements) {
  // Allocate output matrix
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  DiagonalMatrix Omega(num_measurements);

  DiagonalMatrix::DiagonalVectorType &diagonal = Omega.diagonal();

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++)
    diagonal[e] = measurements.second[m].nu;

  for (size_t m = 0; e < num_measurements; e++, m++)
    diagonal[e] = measurements.first[m].tau;

  return Omega;
}

SparseMatrix construct_landmark_and_translational_data_matrix(
    const measurements_t &measurements) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  size_t d = (!measurements.first.empty() ? measurements.first[0].t.size() : 0);

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(d * num_measurements);

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(e, d * measurements.second[m].i + k,
                            -measurements.second[m].l(k));
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(e, d * measurements.first[m].i + k,
                            -measurements.first[m].t(k));
  }

  SparseMatrix T(num_measurements, d * num_poses);
  T.setFromTriplets(triplets.begin(), triplets.end());

  return T;
}

void construct_B_matrices(const measurements_t &measurements, SparseMatrix &B1,
                          SparseMatrix &B2, SparseMatrix &B3) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  // Clear input matrices
  B1.setZero();
  B2.setZero();
  B3.setZero();

  size_t d = (!measurements.first.empty() ? measurements.first[0].t.size() : 0);

  std::vector<Eigen::Triplet<Scalar>> triplets;

  // Useful quantities to cache
  size_t d2 = d * d;
  size_t d3 = d * d * d;

  size_t i, j;  // Indices for the tail and head of the given measurement
  Scalar sqrtnu, sqrttau;
  size_t max_pair;

  /// Construct the matrix B1 from equation (69a) in the tech report
  triplets.reserve(2 * d * num_measurements);

  size_t e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    i = measurements.second[m].i;
    j = measurements.second[m].j;
    sqrtnu = sqrt(measurements.second[m].nu);

    // Block corresponding to the tail of the measurement
    for (size_t l = 0; l < d; l++) {
      triplets.emplace_back(e * d + l, (i + num_landmarks) * d + l,
                            -sqrtnu);  // Diagonal element corresponding to tail
      triplets.emplace_back(e * d + l, j * d + l,
                            sqrtnu);  // Diagonal element corresponding to head
    }
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    i = measurements.first[m].i;
    j = measurements.first[m].j;
    sqrttau = sqrt(measurements.first[m].tau);

    // Block corresponding to the tail of the measurement
    for (size_t l = 0; l < d; l++) {
      triplets.emplace_back(
          e * d + l, (i + num_landmarks) * d + l,
          -sqrttau);  // Diagonal element corresponding to tail
      triplets.emplace_back(e * d + l, (j + num_landmarks) * d + l,
                            sqrttau);  // Diagonal element corresponding to head
    }
  }

  B1.resize(d * num_measurements, d * (num_poses + num_landmarks));
  B1.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B2 from equation (69b) in the tech report
  triplets.clear();
  triplets.reserve(d2 * num_measurements);

  e = 0;

  for (size_t m = 0; e < measurements.second.size(); e++, m++) {
    i = measurements.second[m].i;
    sqrtnu = sqrt(measurements.second[m].nu);
    for (size_t k = 0; k < d; k++)
      for (size_t r = 0; r < d; r++)
        triplets.emplace_back(d * e + r, d2 * i + d * k + r,
                              -sqrtnu * measurements.second[e].l(k));
  }

  for (size_t m = 0; e < num_measurements; e++, m++) {
    i = measurements.first[m].i;
    sqrttau = sqrt(measurements.first[m].tau);
    for (size_t k = 0; k < d; k++)
      for (size_t r = 0; r < d; r++)
        triplets.emplace_back(d * e + r, d2 * i + d * k + r,
                              -sqrttau * measurements.first[m].t(k));
  }

  B2.resize(d * num_measurements, d2 * num_poses);
  B2.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B3 from equation (69c) in the tech report
  triplets.clear();
  triplets.reserve((d3 + d2) * num_measurements);

  e = 0;

  for (size_t m = 0; e < measurements.first.size(); e++, m++) {
    Scalar sqrtkappa = std::sqrt(measurements.first[e].kappa);
    const Matrix &R = measurements.first[e].R;

    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++) {
        i = measurements.first[m].i;  // Tail of measurement
        j = measurements.first[m].j;  // Head of measurement

        // Representation of the -sqrt(kappa) * Rt(i,j) \otimes I_d block
        for (size_t l = 0; l < d; l++)
          triplets.emplace_back(e * d2 + d * r + l, i * d2 + d * c + l,
                                -sqrtkappa * R(c, r));
      }

    for (size_t l = 0; l < d2; l++)
      triplets.emplace_back(e * d2 + l, j * d2 + l, sqrtkappa);
  }

  B3.resize(d2 * measurements.first.size(), d2 * num_poses);
  B3.setFromTriplets(triplets.begin(), triplets.end());
}

SparseMatrix construct_quadratic_form_data_matrix(
    const measurements_t &measurements) {
  const size_t num_measurements =
      measurements.first.size() + measurements.second.size();
  const size_t &num_poses = measurements.num_poses;
  const size_t &num_landmarks = measurements.num_landmarks;

  size_t d = (!measurements.first.empty() ? measurements.first[0].t.size() : 0);

  std::vector<Eigen::Triplet<Scalar>> triplets;

  /// Useful quantities to cache
  size_t d2 = d * d;

  // Number of nonzero elements contributed to Sigma^s by each measurement
  size_t Sigmas_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to U by each measurement
  size_t U_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to N by each measurement
  size_t N_nnz_per_measurement[2] = {0, d};

  // Number of nonzero elements contributed to L(W^c) by each measurement
  size_t LWc_nnz_per_measurement[2] = {4, 0};

  // Number of nonzero elements contributed to Sigma^s by each measurement
  size_t Sigmac_nnz_per_measurement[2] = {0, 1};

  // Number of nonzero elements contributed to V by each measurement
  size_t E_nnz_per_measurement[2] = {2 * d, d};

  // Number of nonzero elements contributed to L(G^z) by each measurement
  size_t LGz_nnz_per_measurement[2] = {2 * d + 2 * d2, 0};

  // Number of nonzero elements contributed to Sigma^z by each measurement
  size_t Sigmaz_nnz_per_measurement[2] = {d2, d2};

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

  for (const SESync::RelativePoseMeasurement &measurement :
       measurements.first) {
    i = measurement.i;  // Tail of measurement
    j = measurement.j;  // Head of measurement

    // Add elements for L(W^tau)
    triplets.emplace_back(i + num_landmarks, i + num_landmarks,
                          measurement.tau);
    triplets.emplace_back(j + num_landmarks, j + num_landmarks,
                          measurement.tau);
    triplets.emplace_back(i + num_landmarks, j + num_landmarks,
                          -measurement.tau);
    triplets.emplace_back(j + num_landmarks, i + num_landmarks,
                          -measurement.tau);

    // Add elements for E (upper-right block)
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(i + num_landmarks,
                            num_poses + i * d + k + num_landmarks,
                            measurement.tau * measurement.t(k));
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(j + num_landmarks,
                            num_poses + i * d + k + num_landmarks,
                            -measurement.tau * measurement.t(k));

    // Add elements for E' (lower-left block)
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + i * d + k + num_landmarks,
                            i + num_landmarks,
                            measurement.tau * measurement.t(k));
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + i * d + k + num_landmarks,
                            j + num_landmarks,
                            -measurement.tau * measurement.t(k));

    // Add elements for L(G^rho)
    // Elements of ith block-diagonal
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + d * i + k + num_landmarks,
                            num_poses + d * i + k + num_landmarks,
                            measurement.kappa);

    // Elements of jth block-diagonal
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + d * j + k + num_landmarks,
                            num_poses + d * j + k + num_landmarks,
                            measurement.kappa);

    // Elements of ij block
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(num_poses + i * d + r + num_landmarks,
                              num_poses + j * d + c + num_landmarks,
                              -measurement.kappa * measurement.R(r, c));

    // Elements of ji block
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(num_poses + j * d + r + num_landmarks,
                              num_poses + i * d + c + num_landmarks,
                              -measurement.kappa * measurement.R(c, r));

    // Add elements for Sigma
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(
            num_poses + i * d + r + num_landmarks,
            num_poses + i * d + c + num_landmarks,
            measurement.tau * measurement.t(r) * measurement.t(c));
  }

  for (const SESync::RelativeLandmarkMeasurement &measurement :
       measurements.second) {
    i = measurement.i;  // Tail of measurement
    j = measurement.j;  // Head of measurement

    // Add elements for Sigma^s
    triplets.emplace_back(j, j, measurement.nu);

    // Add elements for U
    triplets.emplace_back(j, i + num_landmarks, -measurement.nu);

    // Add elements for U'
    triplets.emplace_back(i + num_landmarks, j, -measurement.nu);

    // Add elements for N
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(j, num_poses + i * d + k + num_landmarks,
                            -measurement.nu * measurement.l(k));

    // Add elements for N'
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + i * d + k + num_landmarks, j,
                            -measurement.nu * measurement.l(k));

    // Add elements for Sigma^c
    triplets.emplace_back(i + num_landmarks, i + num_landmarks, measurement.nu);

    // Add elements for E
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(i + num_landmarks,
                            num_poses + i * d + k + num_landmarks,
                            measurement.nu * measurement.l(k));

    // Add elements for E'
    for (size_t k = 0; k < d; k++)
      triplets.emplace_back(num_poses + i * d + k + num_landmarks,
                            i + num_landmarks,
                            measurement.nu * measurement.l(k));

    // Add elements for Sigma^z
    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++)
        triplets.emplace_back(
            num_poses + i * d + r + num_landmarks,
            num_poses + i * d + c + num_landmarks,
            measurement.nu * measurement.l(r) * measurement.l(c));
  }

  SparseMatrix M((d + 1) * num_poses + num_landmarks,
                 (d + 1) * num_poses + num_landmarks);
  M.setFromTriplets(triplets.begin(), triplets.end());

  return M;
}

Matrix chordal_initialization(size_t d, const SparseMatrix &B3) {
  size_t d2 = d * d;
  size_t num_poses = B3.cols() / d2;

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

  SparseMatrix B3red = B3.rightCols((num_poses - 1) * d2);
  // Must be in compressed format to use Eigen::SparseQR!
  B3red.makeCompressed();

  // Vectorization of I_d
  Matrix Id = Matrix::Identity(d, d);
  Eigen::Map<Vector> Id_vec(Id.data(), d2);

  Vector cR = B3.leftCols(d2) * Id_vec;

  Vector rvec;
  Eigen::SPQR<SparseMatrix> QR(B3red);
  rvec = -QR.solve(cR);

  Matrix Rchordal(d, d * num_poses);
  Rchordal.leftCols(d) = Id;
  Rchordal.rightCols((num_poses - 1) * d) =
      Eigen::Map<Matrix>(rvec.data(), d, (num_poses - 1) * d);

  for (size_t i = 1; i < num_poses; i++)
    Rchordal.block(0, i * d, d, d) =
        project_to_SOd(Rchordal.block(0, i * d, d, d));
  return Rchordal;
}

Matrix chordal_initialization(size_t d, const SparseMatrix &B1,
                              const SparseMatrix &B2, const SparseMatrix &B3) {
  const size_t d2 = d * d;
  const size_t num_poses = B3.cols() / d2;
  const size_t size = B1.cols();

  // Vectorization of I_d
  Matrix Id = Matrix::Identity(d, d);
  Eigen::Map<Vector> Id_vec(Id.data(), d2);

  Vector b(B2.rows() + B3.rows());
  b.topRows(B2.rows()) = -B2.leftCols(d2) * Id_vec;
  b.bottomRows(B3.rows()) = -B3.leftCols(d2) * Id_vec;

  std::vector<Eigen::Triplet<Scalar>> triplets;
  triplets.reserve(B1.nonZeros() + B2.nonZeros() + B3.nonZeros());

  for (int k = 0; k < B1.outerSize(); k++) {
    for (SparseMatrix::InnerIterator it(B1, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j < d) continue;

      triplets.emplace_back(i, j - d, value);
    }
  }

  for (int k = 0; k < B2.outerSize(); k++) {
    for (SparseMatrix::InnerIterator it(B2, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j < d2) continue;

      triplets.emplace_back(i, j + size - d2 - d, value);
    }
  }

  for (int k = 0; k < B3.outerSize(); k++) {
    for (SparseMatrix::InnerIterator it(B3, k); it; ++it) {
      const auto &i = it.row();
      const auto &j = it.col();
      const auto &value = it.value();

      if (j < d2) continue;

      triplets.emplace_back(i + B1.rows(), j + size - d2 - d, value);
    }
  }

  SparseMatrix Bred(B1.rows() + B3.rows(), B1.cols() + B3.cols() - d2 - d);

  Bred.setFromTriplets(triplets.begin(), triplets.end());
  Vector rvec;
  Eigen::SPQR<SparseMatrix> QR(Bred);
  rvec = QR.solve(b).tail(d2 * (num_poses - 1));

  Matrix Rchordal(d, d * num_poses);
  Rchordal.leftCols(d) = Id;
  Rchordal.rightCols((num_poses - 1) * d) =
      Eigen::Map<Matrix>(rvec.data(), d, (num_poses - 1) * d);

  for (size_t i = 1; i < num_poses; i++)
    Rchordal.block(0, i * d, d, d) =
        project_to_SOd(Rchordal.block(0, i * d, d, d));

  return Rchordal;
}

Matrix recover_landmarks_and_translations(const SparseMatrix &B1,
                                          const SparseMatrix &B2,
                                          const Matrix &R) {
  size_t d = R.rows();
  size_t n = B1.cols() / d;

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

  // Vectorization of R matrix
  Eigen::Map<Vector> rvec(const_cast<Scalar *>(R.data()), B2.cols());

  // Form the matrix comprised of the right (n-1) block columns of B1
  SparseMatrix B1red = B1.rightCols(d * (n - 1));

  Vector c = B2 * rvec;

  // Solve
  Eigen::SPQR<SparseMatrix> QR(B1red);
  Vector tred = -QR.solve(c);

  // Reshape this result into a d x (n-1) matrix
  Eigen::Map<Matrix> tred_mat(tred.data(), d, n - 1);

  // Allocate output matrix
  Matrix t = Matrix::Zero(d, n);

  // Set rightmost n-1 columns
  t.rightCols(n - 1) = tred_mat;

  return t;
}

Matrix project_to_SOd(const Matrix &M) {
  // Compute the SVD of M
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Scalar detU = svd.matrixU().determinant();
  Scalar detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    return svd.matrixU() * svd.matrixV().transpose();
  } else {
    Matrix Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    return Uprime * svd.matrixV().transpose();
  }
}

Scalar orbit_distance_dS(const Matrix &X, const Matrix &Y, Matrix *G_S) {
  size_t d = X.rows();
  size_t n = X.cols() / d;

  // Compute orbit distance and optimal registration G_S according to Theorem 5
  // in the SE-Sync tech report
  Matrix XYt = X * Y.transpose();

  // Compute singular value decomposition of XY^T
  Eigen::JacobiSVD<Matrix> svd(XYt, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Compute UVt
  Matrix UVt = svd.matrixU() * svd.matrixV().transpose();

  // Construct diagonal matrix Xi = diag(1, ..., 1, det(UV^T))
  Vector Xi_diag = Vector::Constant(d, 1.0);
  // Note that since U and V are orthogonal matrices, then det(UV^T) = +/- 1
  Xi_diag(d - 1) = std::copysign(1.0, UVt.determinant());

  // Compute orbit distance dS
  Scalar dS = sqrt(fabs(
      2 * d * n - 2 * (Xi_diag.array() * svd.singularValues().array()).sum()));

  if (G_S) {
    // Compute optimal registration G_S registering Y to X
    *G_S = svd.matrixU() * Xi_diag.asDiagonal() * svd.matrixV().transpose();
  }
  return dS;
}

Scalar orbit_distance_dO(const Matrix &X, const Matrix &Y, Matrix *G_O) {
  size_t d = X.rows();
  size_t n = X.cols() / d;

  // Compute orbit distance and optimal registration G_O according to Theorem 5
  // in the SE-Sync tech report
  Matrix XYt = X * Y.transpose();

  // Compute singular value decomposition of XY^T
  Eigen::JacobiSVD<Matrix> svd(XYt, Eigen::ComputeFullU | Eigen::ComputeFullV);

  // Compute orbit distance dO
  Scalar dO = sqrt(fabs(2 * d * n - 2 * svd.singularValues().sum()));

  if (G_O) {
    // Compute optimal registration G_O registering Y to X
    *G_O = svd.matrixU() * svd.matrixV().transpose();
  }
  return dO;
}
}  // namespace SESync
