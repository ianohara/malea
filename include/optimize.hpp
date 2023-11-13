#ifndef OPTIMIZE_HPP
#define OPTIMIZE_HPP

#include <iostream>

#include "Eigen/Core"

namespace Vs {
    typedef double ParamType;
    typedef Eigen::Matrix<ParamType, Eigen::Dynamic, 1> ParamVector;

    class AdamOptimizer {
    public:
        AdamOptimizer(
            const double stepsize,
            const double beta_1,
            const double beta_2,
            const double epsilon,
            const size_t param_count) :
            _stepSize(stepsize), _beta_1(beta_1), _beta_2(beta_2), _epsilon(epsilon), _first_moment(param_count),
            _second_moment(param_count), _first_moment_bias_corrected(param_count), _second_moment_bias_corrected(param_count), _steps(0),
            _last_params(param_count) {
                Reset();
            }

        AdamOptimizer() = delete;
        AdamOptimizer(AdamOptimizer&) = delete;

        ParamVector Step(ParamVector current_params, ParamVector current_gradient) {
            _steps++;

            _first_moment = _beta_1 * _first_moment + (1 - _beta_1) * current_gradient;
            _second_moment = _beta_2 * _second_moment.array() + (1 - _beta_2) * current_gradient.array().cwiseProduct(current_gradient.array());

            _first_moment_bias_corrected = _first_moment / (1 - std::pow(_beta_1, _steps));
            _second_moment_bias_corrected = _second_moment / (1 - std::pow(_beta_2, _steps));

            _last_params = current_params.array() - _stepSize *_first_moment_bias_corrected.array() / (_second_moment_bias_corrected.array().sqrt() + _epsilon);

            // std::cout << "After stepping step " << _steps << " opt arrays are..." << std::endl
            //     << "  current_params = " << current_params.transpose() << std::endl
            //     << "  current_gradient = " << current_gradient.transpose() << std::endl
            //     << "  _first_moment = " << _first_moment.transpose() << std::endl
            //     << "  _second_moment = " << _second_moment.transpose() << std::endl
            //     << "  _first_moment_bias_corrected = " << _first_moment_bias_corrected.transpose() << std::endl
            //     << "  _second_moment_bias_corrected = " << _second_moment_bias_corrected.transpose() << std::endl
            //     << "  _last_params = " << _last_params.transpose() << std::endl;

            if (current_gradient.hasNaN()) {
                std::cout << "WARNING: gradient hasNaN in Step" << std::endl;
            }
            return _last_params;
        }

        void Reset() {
            _steps = 0;
            _first_moment.setConstant(0);
            _first_moment_bias_corrected.setConstant(0);
            _second_moment.setConstant(0);
            _second_moment_bias_corrected.setConstant(0);
        }

        size_t GetStepCount() {
            return _steps;
        }

    private:
        const double _stepSize, _beta_1, _beta_2, _epsilon;

        // Note(imo): These are updated in place, so each subsequent Step(...) call assumes
        // that these contain the Step @ t-1 values.
        ParamVector _first_moment, _second_moment,
            _first_moment_bias_corrected, _second_moment_bias_corrected;

        size_t _steps;
        ParamVector _last_params;
    };
}
#endif