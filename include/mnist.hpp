#ifndef MNIST_HPP
#define MNIST_HPP
#include <memory>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "network.hpp"

namespace Vs {
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> EigenImage;

class MNISTLoader {
   public:
    MNISTLoader(const MNISTLoader&) = delete;
    MNISTLoader() = delete;

    MNISTLoader(const std::string& label_path, const std::string& image_path)
        : _label_path(label_path), _image_path(image_path) {
        LoadLabels(label_path);
        LoadImages(image_path);

        _dataset_mean = CalculateMean(_images);
        _dataset_std = CalculateStd(_images);
    }

    void WriteToDirectory(const std::string& path);

    size_t GetPixelsPerImage() { return _images[0].cols() * _images[0].rows(); }

    size_t Count() { return _labels.size(); }

    // NOTE(imo): All the images are stored with pixels as doubles normalized to 0-1
    std::tuple<size_t, EigenImage> GetSample(size_t sample_idx) {
        return std::make_tuple(_labels[sample_idx], _images[sample_idx]);
    }

    double inline GetMean() { return _dataset_mean; }

    double inline GetStd() { return _dataset_std; }

   private:
    const std::string _label_path;
    const std::string _image_path;

    std::vector<char> _labels;
    std::vector<EigenImage> _images;

    double _dataset_mean;
    double _dataset_std;

    void LoadLabels(const std::string& path);
    void LoadImages(const std::string& path);

    static double CalculateMean(std::vector<EigenImage> images);
    static double CalculateStd(std::vector<EigenImage> images);
};

namespace MNIST {
std::shared_ptr<Vs::Network> Network(size_t pixels_per_image);
std::shared_ptr<Vs::Network> MiniNetwork(size_t pixels_per_image);
Vs::IOVector GetOneHotVector(size_t label);
Vs::IOVector ImageToNormalizedInput(Vs::EigenImage image, double dataset_mean, double dataset_std);
}  // namespace MNIST
}  // namespace Vs
#endif /* MNIST_HPP */