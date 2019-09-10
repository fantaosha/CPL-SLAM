/** This file provides a convenient set of utility functions for reading in a
 * set of pose-graph SLAM measurements and constructing the corresponding data
 * matrices used in the CPL-SLAM algorithm.
 *
 * Copyright (C) 2018 - 2019 by Taosha Fan (taosha.fan@gmail.com)
 */

#pragma once

#include <string>

#include <Eigen/Sparse>

#include "CPL-SLAM/CPL-SLAM_types.h"
#include "CPL-SLAM/RelativeLandmarkMeasurement.h"
#include "CPL-SLAM/RelativePoseMeasurement.h"

namespace CPL_SLAM {

struct measurements_t
    : public std::pair<measurements_pose_t, measurements_landmark_t> {
  size_t num_poses = 0;
  size_t num_landmarks = 0;
};

measurements_t read_g2o_file(const std::string &filename);

/** Given a vector of relative pose measurements, this function computes and
 * returns the corresponding rotational connection Laplacian */
ComplexSparseMatrix construct_rotational_connection_Laplacian(
    const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated oriented incidence matrix A */
RealSparseMatrix construct_oriented_incidence_matrix(
    const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated diagonal matrix of landmark and translational
 * measurement precisions */
RealDiagonalMatrix construct_landmark_and_translational_precision_matrix(
    const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated matrix of raw translational measurements */
ComplexSparseMatrix construct_landmark_and_translational_data_matrix(
    const measurements_t &measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the B matrices defined in equation (69) of the tech report */
void construct_B_matrices(const measurements_t &measurements,
                          ComplexSparseMatrix &B1, ComplexSparseMatrix &B2,
                          ComplexSparseMatrix &B3);

/** Given a vector of relative pose measurements, this function constructs the
 * matrix M parameterizing the objective in the translation-explicit formulation
 * of the CPL-SLAM problem (Problem 2) in the CPL-SLAM tech report) */
ComplexSparseMatrix construct_quadratic_form_data_matrix(
    const measurements_t &measurements);

/** Given the measurement matrix B3 defined in equation (69c) of the tech report
 * and the problem dimension d, this function computes and returns the
 * corresponding chordal initialization for the rotational states */
ComplexVector chordal_initialization(const ComplexSparseMatrix &B3);

/** Given the measurement matrix B1, B2, B3 defined in equation (69) of the
 * tech report and the problem dimension d, this function computes and returns
 * the corresponding chordal initialization for the rotational states */
ComplexVector chordal_initialization(const ComplexSparseMatrix &B1,
                                     const ComplexSparseMatrix &B2,
                                     const ComplexSparseMatrix &B3);

/** Given the measurement matrices B1 and B2 and a matrix R of rotational state
 * estimates, this function computes and returns the corresponding optimal
 * translation estimates */
ComplexVector recover_landmarks_and_translations(const ComplexSparseMatrix &B1,
                                                 const ComplexSparseMatrix &B2,
                                                 const ComplexVector &R);
}  // namespace CPL_SLAM
