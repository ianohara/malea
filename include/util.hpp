#ifndef UTIL_HPP
#define UTIL_HPP
#include <random>

namespace Vs {
namespace Util {
    template<typename T>
    T RandInRange(T min, T max) {
            static_assert(std::is_same<T, double>::value || std::is_same<T, float>::value,
            "Can only give RandInRange for double and float");
        return (min + (max - min) * (std::rand() / static_cast<T>(RAND_MAX)));
    }

    template<typename T>
    T RandInGaussian(T mean, T std) {
        auto box_mueller = []() -> T {
            T rand_1 = RandInRange(0.0, 1.0);
            T rand_2 = RandInRange(0.0, 1.0);

            return std::sqrt(-2 * std::log(rand_1)) * std::cos(M_2_PI * rand_2);
        };

        return mean + std * box_mueller();
    }
}
}
#endif /* UTIL_HPP */