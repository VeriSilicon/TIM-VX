#ifndef TIM_VX_TEST_UTILS_H_
#define TIM_VX_TEST_UTILS_H_

#include <cmath>
#include <limits>
#include <ostream>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"

// A gmock matcher that check that elements of a float vector match to a given
// tolerance.
inline std::vector<::testing::Matcher<float>> ArrayFloatNear(
    const std::vector<float>& values, float max_abs_error = 1e-5) {
  std::vector<::testing::Matcher<float>> matchers;
  matchers.reserve(values.size());
  for (const float& v : values) {
    matchers.emplace_back(testing::FloatNear(v, max_abs_error));
  }
  return matchers;
}

template <typename T>
std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
  int32_t zero_point = 0;
  float scale = 0;
  const T qmin = std::numeric_limits<T>::min();
  const T qmax = std::numeric_limits<T>::max();
  const float qmin_double = qmin;
  const float qmax_double = qmax;
  // 0 should always be a representable value. Let's assume that the initial
  // min,max range contains 0.
  if (f_min == f_max) {
    // Special case where the min,max range is a point. Should be {0}.
    return {scale, zero_point};
  }

  // General case.
  //
  // First determine the scale.
  scale = (f_max - f_min) / (qmax_double - qmin_double);

  // Zero-point computation.
  // First the initial floating-point computation. The zero-point can be
  // determined from solving an affine equation for any known pair
  // (real value, corresponding quantized value).
  // We know two such pairs: (rmin, qmin) and (rmax, qmax).
  // The arithmetic error on the zero point computed from either pair
  // will be roughly machine_epsilon * (sum of absolute values of terms)
  // so we want to use the variant that adds the smaller terms.
  const float zero_point_from_min = qmin_double - f_min / scale;
  const float zero_point_from_max = qmax_double - f_max / scale;

  const float zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(f_min / scale);

  const float zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(f_max / scale);

  const float zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  // Now we need to nudge the zero point to be an integer
  // (our zero points are integer, and this is motivated by the requirement
  // to be able to represent the real value "0" exactly as a quantized value,
  // which is required in multiple places, for example in Im2col with SAME
  //  padding).

  T nudged_zero_point = 0;
  if (zero_point_double < qmin_double) {
    nudged_zero_point = qmin;
  } else if (zero_point_double > qmax_double) {
    nudged_zero_point = qmax;
  } else {
    nudged_zero_point = static_cast<T>(std::round(zero_point_double));
  }

  // The zero point should always be in the range of quantized value,
  // // [qmin, qmax].

  zero_point = nudged_zero_point;
  // finally, return the values
  return {scale, zero_point};
}

template <typename T>
inline std::vector<T> Quantize(const std::vector<float>& data, float scale,
                               int32_t zero_point) {
  std::vector<T> q;
  for (const auto& f : data) {
    q.push_back(static_cast<T>(std::max<float>(
        std::numeric_limits<T>::min(),
        std::min<float>(std::numeric_limits<T>::max(),
                        std::round(zero_point + (f / scale))))));
  }
  return q;
}

template <typename T>
inline std::vector<float> Dequantize(const std::vector<T>& data, float scale,
                                     int32_t zero_point) {
  std::vector<float> f;
  f.reserve(data.size());
  for (const T& q : data) {
    f.push_back(scale * (q - zero_point));
  }
  return f;
}


#endif /* TIM_VX_TEST_UTILS_H_ */