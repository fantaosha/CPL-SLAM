/** A lightweight struct encapsulating a single relative pose-graph SLAM
 * measurement sampled from the generative model using the complex number
 * representation.
 *
 * Copyright (C) 2016 - 2018 by David M. Rosen (dmrosen@mit.edu)
 */

#pragma once

#include <Eigen/Dense>
#include <iostream>

#include "CPL-SLAM/CPL-SLAM_types.h"

namespace CPL_SLAM {

/** A simple struct that contains the elements of a relative pose measurement */
struct RelativeLandmarkMeasurement {
  /** 0-based index of first pose */
  size_t i;

  /** 0-based index of second pose */
  size_t j;

  /** Landmark measurement */
  Complex l;

  /** Measurement precision */
  Scalar nu;

  /** Simple default constructor; does nothing */
  RelativeLandmarkMeasurement() {}

  /** Basic constructor */
  RelativeLandmarkMeasurement(size_t pose, size_t landmark,
                              Complex const &relative_position,
                              Scalar precision)
      : i(pose), j(landmark), l(relative_position), nu(precision) {}

  /** A utility function for streaming Nodes to cout */
  inline friend std::ostream &operator<<(
      std::ostream &os, const RelativeLandmarkMeasurement &measurement) {
    os << "i: " << measurement.i << std::endl;
    os << "j: " << measurement.j << std::endl;
    os << "l: " << std::endl << measurement.l << std::endl;
    os << "nu: " << measurement.nu << std::endl;

    return os;
  }
};

/** Typedef for a vector of RelativeLandmarkMeasurements */
typedef std::vector<CPL_SLAM::RelativeLandmarkMeasurement> measurements_landmark_t;
}  // namespace CPL_SLAM
