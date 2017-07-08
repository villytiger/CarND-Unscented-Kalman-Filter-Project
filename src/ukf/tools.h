#ifndef EKF_TOOLS_H_
#define EKF_TOOLS_H_

#include "Eigen/Dense"

template <typename Container, typename Type = typename Container::value_type>
Type CalculateRmse(const Container& estimations,
                   const Container& ground_truth) {
  assert(!estimations.empty() && estimations.size() == ground_truth.size());

  Type rmse =
      Type::Zero(estimations.front().rows(), estimations.front().cols());
  size_t size = 0;

  auto it1 = estimations.begin();
  auto it2 = ground_truth.begin();
  for (; it1 != estimations.end(); ++it1, ++it2) {
    assert(it1->size() == it2->size());

    rmse += (*it1 - *it2).pow(2);
    ++size;
  }

  return (rmse / size).sqrt();
}

#endif  // EKF_TOOLS_H_
