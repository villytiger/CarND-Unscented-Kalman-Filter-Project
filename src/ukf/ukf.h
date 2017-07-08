#ifndef UKF_UKF_H
#define UKF_UKF_H

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "Eigen/Dense"

#include "ukf/measurement.h"

namespace ukf {

constexpr const int kStateSize = 5;

constexpr const int kAug = 7;

constexpr const int kLambda = 3 - kAug;

constexpr const int kSigmaSize = kAug * 2 + 1;

constexpr const int kLidarStateSize = 2;

constexpr const int kRadarStateSize = 3;

// [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
typedef Eigen::Matrix<double, kStateSize, 1> State;

typedef Eigen::Matrix<double, kStateSize, kStateSize> Covariance;

typedef Eigen::Matrix<double, kStateSize, kSigmaSize> Sigma;

typedef Eigen::Matrix<double, kAug, kSigmaSize> AugSigma;

typedef Eigen::Matrix<double, kLidarStateSize, 1> LidarState;

typedef Eigen::Matrix<double, kLidarStateSize, kLidarStateSize> LidarCovariance;

typedef Eigen::Matrix<double, kLidarStateSize, kSigmaSize> LidarSigma;

typedef Eigen::Matrix<double, kRadarStateSize, 1> RadarState;

typedef Eigen::Matrix<double, kRadarStateSize, kRadarStateSize> RadarCovariance;

typedef Eigen::Matrix<double, kRadarStateSize, kSigmaSize> RadarSigma;

template <typename Measurement>
class Helper;

template <>
class Helper<LidarMeasurement> {
 public:
  static Eigen::Vector2d Convert(const LidarState& z) { return z; }

  Helper(const LidarState& z, const Sigma& Xsig) : z_(z), Xsig_(Xsig) {}

  void Update(State* x, Covariance* P);

 private:
  std::tuple<LidarSigma, LidarState, LidarCovariance> PredictMeasurement();

  LidarState z_;
  Sigma Xsig_;
};

template <>
class Helper<RadarMeasurement> {
 public:
  static Eigen::Vector2d Convert(const RadarState& z) {
    const auto rho = z(0);
    const auto phi = z(1);
    return {rho * cos(phi), rho * sin(phi)};
  }

  Helper(const RadarState& z, const Sigma& Xsig) : z_(z), Xsig_(Xsig) {}

  void Update(State* x, Covariance* P);

 private:
  std::tuple<RadarSigma, RadarState, RadarCovariance> PredictMeasurement();

  RadarState z_;
  Sigma Xsig_;
};

class Ukf {
 public:
  template <typename Measurement>
  void ProcessMeasurement(const Measurement& measurement) {
    if (!is_initialized_) {
      x_.head(2) = Helper<Measurement>::Convert(measurement.data);

      previous_timestamp_ = measurement.timestamp;
      is_initialized_ = true;

      return;
    }

    if (Enabled<Measurement>()) {
      auto Xsig =
          Predict((measurement.timestamp - previous_timestamp_) / 1000000.0);

      Helper<Measurement> helper(measurement.data, Xsig);
      helper.Update(&x_, &P_);

      previous_timestamp_ = measurement.timestamp;
    }
  }

  Eigen::Vector4d Estimate();

 private:
  Sigma Predict(double delta);

  Sigma PredictSigmaPoints(double delta);

  AugSigma GenerateSigmaPoints();

  template <typename Measurement>
  bool Enabled();

  bool is_initialized_ = false;

  State x_ = State::Zero();

  Covariance P_ = Covariance::Identity();

  uint_fast64_t previous_timestamp_ = 0;
};

}  // namespace ukf

#endif  // UKF_UKF_H
