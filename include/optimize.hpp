#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include "Eigen/Core"

namespace Vs {
    template <int ParamCount> class AdamOptimizer {
    public:
        typedef double ParamType;
        typedef Eigen::Matrix<ParamType, ParamCount, 1> ParamVector;

        AdamOptimizer(
            const double stepsize,
            const double beta_1,
            const double beta_2,
            const double epsilon) :
            _stepSize(stepsize), _beta_1(beta_1), _beta_2(beta_2), _epsilon(epsilon), _first_moment(0),
            _second_moment(0), _first_moment_bias_corrected(0), _second_moment_bias_corrected(0), _steps(0) {}

        ParamVector Step(ParamVector current_params, ParamVector current_gradient) {
            _steps++;

            _first_moment = _beta_1 * _first_moment + (1 - _beta_1) * current_gradient;
            _second_moment = _beta_2 * _second_moment + (1 - _beta_2) * current_gradient.array().cwiseProduct(current_gradient);

            _first_moment_bias_corrected = _first_moment / (1 - std::pow(_beta_1, _steps));
            _second_moment_bias_corrected = _second_moment / (1 - std::pow(_beta_2, _steps));

            return current_params - _stepSize * _second_moment_bias_corrected / (_first_moment_bias_corrected.array().sqrt() + _epsilon);
        }

    private:
        const double _stepSize, _beta_1, _beta_2, _epsilon;

        // Note(imo): These are updated in place, so each subsequent Step(...) call assumes
        // that these contain the Step @ t-1 values.
        Eigen::Matrix<ParamType, ParamCount, 1> _first_moment, _second_moment,
            _first_moment_bias_corrected, _second_moment_bias_corrected;

        unsigned int _steps;
    };
}
#endif