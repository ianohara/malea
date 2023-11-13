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

    size_t GetPixelsPerImage() {
        return _images[0].cols() * _images[0].rows();
    }

    size_t Count() {
        return _labels.size();
    }

    std::tuple<size_t, EigenImage> GetSample(size_t sample_idx) {
        return std::make_tuple(_labels[sample_idx], _images[sample_idx]);
    }

    private:
        const std::string _label_path;
        const std::string _image_path;

        std::vector<char> _labels;
        std::vector<EigenImage> _images;

        void LoadLabels(const std::string& path);
        void LoadImages(const std::string& path);
    };
    namespace MNIST {
        std::shared_ptr<Vs::Network> Network(size_t pixels_per_image);
        Vs::IOVector GetOneHotVector(size_t label);
        Vs::IOVector ImageToInput(Vs::EigenImage image);
    }
}
#endif /* MNIST_HPP */