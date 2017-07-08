#ifndef UKF_MEASUREMENT_H_
#define UKF_MEASUREMENT_H_

#include "Eigen/Dense"

struct LidarMeasurement {
  uint_fast64_t timestamp;
  Eigen::Vector2d data;
};

struct RadarMeasurement {
  uint_fast64_t timestamp;
  Eigen::Vector3d data;
};

#endif  // UKF_MEASUREMENT_H_
