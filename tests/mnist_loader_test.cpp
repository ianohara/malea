#include "gtest/gtest.h"

#include "mnist.hpp"

#include <cstdlib>
#include <tuple>
#include <fstream>

TEST(MNISTLoader, LoadingWorks) {
    std::string mnist_labels_path = "", mnist_images_path = "";
    if (std::getenv("VS_MNIST_LABELS_PATH")) {
        mnist_labels_path = std::string(std::getenv("VS_MNIST_LABELS_PATH"));
    }

    if (std::getenv("VS_MNIST_IMAGES_PATH")) {
        mnist_images_path = std::string(std::getenv("VS_MNIST_IMAGES_PATH"));
    }


    if (mnist_labels_path == "") {
        std::string possible_label_build_path = VS_CMAKE_BUILD_ROOT "/data/train-labels.idx1-ubyte";
        std::cout << "Checking '" << possible_label_build_path << "'" << std::endl;
        std::ifstream check_label_path_in_build(possible_label_build_path);
        if (check_label_path_in_build.good()) {
            mnist_labels_path = possible_label_build_path;
        }
    }

    if (mnist_images_path == "") {
        std::string possible_images_build_path = VS_CMAKE_BUILD_ROOT "/data/train-images.idx3-ubyte";
        std::cout << "Checking '" << possible_images_build_path << "'" << std::endl;
        std::ifstream check_images_path_in_build(possible_images_build_path);
        if (check_images_path_in_build.good()) {
            mnist_images_path = possible_images_build_path;
        }
    }

    if (mnist_labels_path == "" || mnist_images_path == "") {
        std::cout << "Both VS_MNIST_LABELS_PATH and VS_MNIST_IMAGES_PATH must be defined to run the MNIST Loader tests." << std::endl;
        return;
    }

    Vs::MNISTLoader loader{std::string(mnist_labels_path), std::string(mnist_images_path)};
}