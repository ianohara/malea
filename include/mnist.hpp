#ifndef MNIST_HPP
#define MNIST_HPP
#include <string>
#include <vector>
#include <memory>

#include "Eigen/Core"

#include "network.hpp"

namespace Vs {
    typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> EigenImage;

    class MNISTLoader {
    public:
    MNISTLoader(const MNISTLoader&) = delete;
    MNISTLoader() = delete;

    MNISTLoader(const std::string& label_path, const std::string& image_path) : _label_path(label_path), _image_path(image_path) {
        LoadLabels(label_path);
        LoadImages(image_path);
    }

    void WriteToDirectory(const std::string& path);

    private:
        const std::string _label_path;
        const std::string _image_path;

        std::vector<char> _labels;
        std::vector<EigenImage> _images;

        void LoadLabels(const std::string& path);
        void LoadImages(const std::string& path);
    };

    std::shared_ptr<Network> MNISTNetwork(size_t pixels_per_image);
}
#endif /* MNIST_HPP */