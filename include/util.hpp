#ifndef UTIL_HPP
#define UTIL_HPP

#include <mutex>
#include <random>

#include "Eigen/Core"

namespace Vs {
static constexpr bool Debug = false;

typedef double FVal;
typedef double BVal;
typedef double WVal;

typedef Eigen::Matrix<FVal, Eigen::Dynamic, 1> IOVector;
typedef Eigen::Matrix<BVal, Eigen::Dynamic, 1> GradVector;
}  // namespace Vs

namespace Vs {
namespace Util {
static std::mutex rand_in_range;

template <typename T>
T RandInRange(T min, T max) {
    static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
                  "Can only give RandInRange for double and float");

    {
        std::lock_guard<std::mutex> lg(rand_in_range);
        T rand_val = (std::rand() / static_cast<T>(RAND_MAX));
        T final_value = (min + (max - min) * rand_val);
        return final_value;
    }
}

template <typename T>
T RandInGaussian(T mean, T std) {
    auto box_mueller = []() -> T {
        T rand_1 = RandInRange(0.0, 1.0);
        T rand_2 = RandInRange(0.0, 1.0);

        return std::sqrt(-2 * std::log(rand_1)) * std::cos(2 * M_PI * rand_2);
    };

    auto norm_dist_val = box_mueller();
    T final_val = mean + std * box_mueller();
    return final_val;
}

template <typename T, int R, int C>
void TopTenStream(std::ostream &os, Eigen::Matrix<T, R, C> m) {
    const size_t count = 10;
    std::vector<std::tuple<T, std::string>> vals_and_coords;
    for (size_t row = 0; row < m.rows(); row++) {
        for (size_t col = 0; col < m.cols(); col++) {
            std::stringstream ss;
            ss << "(" << row << "," << col << ")";
            vals_and_coords.push_back(std::make_tuple(m(row, col), ss.str()));
        }
    }

    std::sort(vals_and_coords.rbegin(), vals_and_coords.rend(),
              [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });

    for (size_t idx = 0; idx < count; idx++) {
        auto [val, pos] = vals_and_coords[idx];
        os << pos << ": " << val << ", ";
    }
}

template <typename T, int R, int C>
void CompareTopTen(std::ostream &os, Eigen::Matrix<T, R, C> &m, Eigen::Matrix<T, R, C> &other) {
    const size_t count = 10;
    std::vector<std::tuple<T, std::string, T>> vals_and_coords;
    for (size_t row = 0; row < m.rows(); row++) {
        for (size_t col = 0; col < m.cols(); col++) {
            std::stringstream ss;
            ss << "(" << row << "," << col << ")";
            vals_and_coords.push_back(std::make_tuple(m(row, col), ss.str(), other(row, col)));
        }
    }

    std::sort(vals_and_coords.rbegin(), vals_and_coords.rend(),
              [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });

    for (size_t idx = 0; idx < count; idx++) {
        auto [val, pos, other_val] = vals_and_coords[idx];
        os << pos << ": " << val << " vs " << other_val << " diff= " << val - other_val << std::endl;
    }
}

template <typename T, int R, int C>
void TopTenDifferences(std::ostream &os, Eigen::Matrix<T, R, C> &m, Eigen::Matrix<T, R, C> &other,
                       std::function<std::string(size_t, size_t)> describer) {
    const size_t count = 90;
    std::vector<std::tuple<T, std::string, T>> vals_and_coords;
    for (size_t row = 0; row < m.rows(); row++) {
        for (size_t col = 0; col < m.cols(); col++) {
            vals_and_coords.push_back(std::make_tuple(m(row, col), describer(row, col), other(row, col)));
        }
    }

    std::sort(vals_and_coords.rbegin(), vals_and_coords.rend(), [](auto a, auto b) {
        return std::abs(std::get<0>(a) - std::get<2>(a)) < std::abs(std::get<0>(b) - std::get<2>(b));
    });

    for (size_t idx = 0; idx < count && idx < vals_and_coords.size(); idx++) {
        auto [val, pos, other_val] = vals_and_coords[idx];
        os << pos << ": " << val << " vs " << other_val << " diff= " << val - other_val << std::endl;
    }
}
template <typename T, int R>
void DifferencesForIndicies(std::ostream &os, const std::vector<size_t> &indicies, Eigen::Matrix<T, R, 1> &m,
                            Eigen::Matrix<T, R, 1> &other, std::function<std::string(size_t, size_t)> describer) {
    for (size_t idx : indicies) {
        T val = m(idx);
        T other_val = other(idx);
        std::string pos = describer(idx, 0);
        os << pos << ": " << val << " vs " << other_val << " diff= " << val - other_val << std::endl;
    }
}
}  // namespace Util
}  // namespace Vs
#endif /* UTIL_HPP */