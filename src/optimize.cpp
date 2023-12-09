#include "optimize.hpp"

#include <iostream>

#include "Eigen/Core"
#include "cxxopts.hpp"

namespace Ml {
        ParamVector AdamOptimizer::Step(const ParamVector& current_params, const ParamVector& current_gradient) {
        _steps++;

        _first_moment = _beta_1 * _first_moment + (1 - _beta_1) * current_gradient;
        _second_moment = _beta_2 * _second_moment.array() +
                         (1 - _beta_2) * current_gradient.array().cwiseProduct(current_gradient.array());

        _first_moment_bias_corrected = _first_moment / (1 - std::pow(_beta_1, _steps));
        _second_moment_bias_corrected = _second_moment / (1 - std::pow(_beta_2, _steps));

        _last_params = current_params.array() - _stepSize * _first_moment_bias_corrected.array() /
                                                    (_second_moment_bias_corrected.array().sqrt() + _epsilon);

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

    void AdamOptimizer::Reset() {
        _steps = 0;
        _first_moment.setConstant(0);
        _first_moment_bias_corrected.setConstant(0);
        _second_moment.setConstant(0);
        _second_moment_bias_corrected.setConstant(0);
    }
} // namespace Ml