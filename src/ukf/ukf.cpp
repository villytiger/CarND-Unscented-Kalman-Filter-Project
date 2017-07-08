#include "ukf/ukf.h"

#include <iostream>

#include "Eigen/Dense"

using namespace std;

using Eigen::Matrix;

namespace {

auto kWeights = []() {
  Matrix<double, ukf::kSigmaSize, 1> weights;

  weights(0) = 1.0 * ukf::kLambda / (ukf::kLambda + ukf::kAug);

  for (int i = 1; i < weights.size(); i++) {
    weights(i) = 0.5 / (ukf::kAug + ukf::kLambda);
  }

  return weights;
}();

void normalize(double* angle) {
  while (*angle > M_PI) *angle -= 2. * M_PI;
  while (*angle < -M_PI) *angle += 2. * M_PI;
}

ukf::State sub(const ukf::State& state1, const ukf::State& state2) {
  ukf::State state = state1 - state2;
  normalize(&state(3));
  return state;
}

ukf::RadarState sub(const ukf::RadarState& state1,
                    const ukf::RadarState& state2) {
  ukf::RadarState state = state1 - state2;
  normalize(&state(1));
  return state;
}

double div(double value1, double value2) {
  if (value2) return value1 / value2;
  else return value1 / std::numeric_limits<double>::min();
}

}  // namespace

namespace ukf {

void Helper<LidarMeasurement>::Update(State* x, Covariance* P) {
  LidarSigma Zsig;
  LidarState z;
  LidarCovariance S;
  std::tie(Zsig, z, S) = PredictMeasurement();

  auto Tc = Matrix<double, kStateSize, kLidarStateSize>::Zero().eval();
  for (int i = 0; i < Zsig.cols(); ++i) {
    LidarState z_diff = Zsig.col(i) - z;
    State x_diff = sub(Xsig_.col(i), *x);

    Tc = Tc + kWeights(i) * x_diff * z_diff.transpose();
  }

  const auto K = Tc * S.inverse();

  LidarState z_diff = z_ - z;

  *x = *x + K * z_diff;
  *P = *P - K * S * K.transpose();
}

tuple<LidarSigma, LidarState, LidarCovariance>
Helper<LidarMeasurement>::PredictMeasurement() {
  LidarSigma Zsig;

  for (int i = 0; i < Xsig_.cols(); ++i) {
    Zsig(0, i) = Xsig_(0, i);
    Zsig(1, i) = Xsig_(1, i);
  }

  auto z_pred = LidarState::Zero().eval();
  for (int i = 0; i < kWeights.size(); ++i) {
    z_pred = z_pred + kWeights(i) * Zsig.col(i);
  }

  auto S = LidarCovariance::Zero().eval();
  for (int i = 0; i < Zsig.cols(); ++i) {
    LidarState z_diff = Zsig.col(i) - z_pred;

    S = S + kWeights(i) * z_diff * z_diff.transpose();
  }

  LidarCovariance R;
  // clang-format off
  R <<
    // Laser measurement noise standard deviation position1 in m
    pow(0.15, 2), 0,
    // Laser measurement noise standard deviation position2 in m
    0, pow(0.15, 2);
  // clang-format on
  S = S + R;

  return make_tuple(Zsig, z_pred, S);
}

void Helper<RadarMeasurement>::Update(State* x, Covariance* P) {
  RadarSigma Zsig;
  RadarState z;
  RadarCovariance S;
  std::tie(Zsig, z, S) = PredictMeasurement();

  auto Tc = Matrix<double, kStateSize, kRadarStateSize>::Zero().eval();

  for (int i = 0; i < Zsig.cols(); ++i) {
    RadarState z_diff = sub(Zsig.col(i), z);
    State x_diff = sub(Xsig_.col(i), *x);

    Tc += kWeights(i) * x_diff * z_diff.transpose();
  }

  const auto K = Tc * S.inverse();
  RadarState z_diff = sub(z_, z);

  *x += K * z_diff;
  *P -= K * S * K.transpose();
}

tuple<RadarSigma, RadarState, RadarCovariance>
Helper<RadarMeasurement>::PredictMeasurement() {
  RadarSigma Zsig;

  for (int i = 0; i < Xsig_.cols(); ++i) {
    double px = Xsig_(0, i);
    double py = Xsig_(1, i);
    double v = Xsig_(2, i);
    double yaw = Xsig_(3, i);

    double vx = cos(yaw) * v;
    double vy = sin(yaw) * v;

    Zsig(0, i) = sqrt(px * px + py * py);
    Zsig(1, i) = px ? atan2(py, px) : atan2(py, std::numeric_limits<double>::min());
    Zsig(2, i) = div(px * vx + py * vy, sqrt(px * px + py * py));
  }

  auto z_pred = RadarState::Zero().eval();
  for (int i = 0; i < Zsig.cols(); ++i) {
    z_pred += kWeights(i) * Zsig.col(i);
  }

  auto S = RadarCovariance::Zero().eval();
  for (int i = 0; i < Zsig.cols(); ++i) {
    RadarState z_diff = sub(Zsig.col(i), z_pred);

    S += kWeights(i) * z_diff * z_diff.transpose();
  }

  RadarCovariance R;
  // clang-format off
  R <<
    // Radar measurement noise standard deviation radius in m
    pow(0.3, 2), 0, 0,
    // Radar measurement noise standard deviation angle in rad
    0, pow(0.03, 2), 0,
    // Radar measurement noise standard deviation radius change in m/s
    0, 0, pow(0.3, 2);
  // clang-format on
  S += R;

  return make_tuple(Zsig, z_pred, S);
}

Sigma Ukf::Predict(double delta) {
  auto Xsig = PredictSigmaPoints(delta);

  x_.setZero();

  for (int i = 0; i < Xsig.cols(); ++i) {
    x_ += kWeights(i) * Xsig.col(i);
  }

  P_.setZero();

  for (int i = 0; i < Xsig.cols(); ++i) {
    State x_diff = sub(Xsig.col(i), x_);

    P_ += kWeights(i) * x_diff * x_diff.transpose();
  }

  return Xsig;
}

Eigen::Vector4d Ukf::Estimate() {
  const auto v = x_(2);
  const auto psi = x_(3);

  Eigen::Vector4d data;
  data << x_(0), x_(1), v * cos(psi), v * sin(psi);

  return data;
}

Sigma Ukf::PredictSigmaPoints(double delta) {
  Sigma Xsig;

  auto Xsig_aug = GenerateSigmaPoints();

  for (int i = 0; i < Xsig_aug.cols(); ++i) {
    const auto px = Xsig_aug(0, i);
    const auto py = Xsig_aug(1, i);
    const auto v = Xsig_aug(2, i);
    const auto yaw = Xsig_aug(3, i);
    const auto yawd = Xsig_aug(4, i);
    const auto nu_a = Xsig_aug(5, i);
    const auto nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    if (fabs(yawd) > numeric_limits<double>::min()) {
      px_p = px + div(v, yawd) * (sin(yaw + yawd * delta) - sin(yaw));
      py_p = py + div(v, yawd) * (cos(yaw) - cos(yaw + yawd * delta));
    } else {
      px_p = px + v * delta * cos(yaw);
      py_p = py + v * delta * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd * delta;
    double yawd_p = yawd;

    px_p = px_p + 0.5 * nu_a * delta * delta * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta * delta * sin(yaw);
    v_p = v_p + nu_a * delta;

    yaw_p = yaw_p + 0.5 * nu_yawdd * delta * delta;
    yawd_p = yawd_p + nu_yawdd * delta;

    Xsig(0, i) = px_p;
    Xsig(1, i) = py_p;
    Xsig(2, i) = v_p;
    Xsig(3, i) = yaw_p;
    Xsig(4, i) = yawd_p;
  }

  return Xsig;
}

AugSigma Ukf::GenerateSigmaPoints() {
  Matrix<double, kAug, 1> x_aug;
  x_aug.head(kStateSize) = x_;
  x_aug(kStateSize) = 0;
  x_aug(kStateSize + 1) = 0;

  auto P_aug = Matrix<double, kAug, kAug>::Zero().eval();
  P_aug.topLeftCorner(kStateSize, kStateSize) = P_;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  P_aug(kStateSize, kStateSize) = pow(0.5, 2);
  // Process noise standard deviation yaw acceleration in rad/s^2
  P_aug(kStateSize + 1, kStateSize + 1) = pow(0.5, 2);

  Matrix<double, kAug, kAug> L = P_aug.llt().matrixL();

  AugSigma Xsig;
  Xsig.col(0) = x_aug;

  for (int i = 0; i < L.cols(); ++i) {
    Xsig.col(i + 1) = x_aug + sqrt(kLambda + kAug) * L.col(i);
    Xsig.col(i + 1 + kAug) = x_aug - sqrt(kLambda + kAug) * L.col(i);
  }

  return Xsig;
}

template <>
bool Ukf::Enabled<RadarMeasurement>() {
  return true;
}

template <>
bool Ukf::Enabled<LidarMeasurement>() {
  return true;
}

}  // namespace ukf
