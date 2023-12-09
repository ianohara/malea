#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <iostream>

#include "Eigen/Core"

namespace Ml {
typedef double ParamType;
typedef Eigen::Matrix<ParamType, Eigen::Dynamic, 1> ParamVector;

class AdamOptimizer {
   public:
    AdamOptimizer(const double stepsize, const double beta_1, const double beta_2, const double epsilon,
                  const size_t param_count)
        : _stepSize(stepsize),
          _beta_1(beta_1),
          _beta_2(beta_2),
          _epsilon(epsilon),
          _first_moment(param_count),
          _second_moment(param_count),
          _first_moment_bias_corrected(param_count),
          _second_moment_bias_corrected(param_count),
          _steps(0),
          _last_params(param_count) {
        Reset();
    }

    AdamOptimizer() = delete;
    AdamOptimizer(AdamOptimizer&) = delete;

    ParamVector Step(const ParamVector& current_params, const ParamVector& current_gradient);

    void Reset();

    size_t GetStepCount() { return _steps; }

   private:
    const double _stepSize, _beta_1, _beta_2, _epsilon;

    // Note(imo): These are updated in place, so each subsequent Step(...) call assumes
    // that these contain the Step @ t-1 values.
    ParamVector _first_moment, _second_moment, _first_moment_bias_corrected, _second_moment_bias_corrected;

    size_t _steps;
    ParamVector _last_params;
};
}  // namespace Ml
#endif