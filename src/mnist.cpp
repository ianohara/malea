#include "mnist.hpp"

#include <endian.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace Vs {
// Load the labels file into the internal labels (0-9 values only) vector.
//
// The layout of the data is taken from: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
void MNISTLoader::LoadLabels(const std::string &path) {
    std::ifstream in_file(path, std::ios::binary | std::ios::in);
    if (in_file.is_open()) {
        uint32_t first_big_endian_int;
        uint32_t second_big_endian_int;
        in_file.read(reinterpret_cast<char *>(&first_big_endian_int), sizeof(first_big_endian_int));
        in_file.read(reinterpret_cast<char *>(&second_big_endian_int), sizeof(second_big_endian_int));

        int32_t magic = static_cast<int32_t>(be32toh(first_big_endian_int));
        int32_t size = static_cast<int32_t>(be32toh(second_big_endian_int));

        if (magic != 2049) {
            throw std::runtime_error("Invalid magic number in label file");
        }

        std::copy(std::istreambuf_iterator<char>(in_file), std::istreambuf_iterator<char>(),
                  std::back_inserter(_labels));

        assert(static_cast<int>(_labels.size()) == size);
    } else {
        throw std::runtime_error(path);
    }
}

// Load the images file into our internal vector of eigen matrixies (each is an image).
//
// The layout of data is taken from: https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
void MNISTLoader::LoadImages(const std::string &path) {
    std::ifstream in_file(path, std::ios::binary | std::ios::in);
    if (in_file.is_open()) {
        uint32_t first_be, second_be, third_be, fourth_be;
        in_file.read(reinterpret_cast<char *>(&first_be), sizeof(first_be));
        in_file.read(reinterpret_cast<char *>(&second_be), sizeof(second_be));
        in_file.read(reinterpret_cast<char *>(&third_be), sizeof(third_be));
        in_file.read(reinterpret_cast<char *>(&fourth_be), sizeof(fourth_be));

        int32_t magic = static_cast<int32_t>(be32toh(first_be));
        int32_t size = static_cast<int32_t>(be32toh(second_be));
        int32_t rows = static_cast<int32_t>(be32toh(third_be));
        int32_t cols = static_cast<int32_t>(be32toh(fourth_be));

        if (magic != 2051) {
            throw std::runtime_error("Invalid magic number in label file");
        }

        for (int image_idx = 0; image_idx < size; image_idx++) {
            Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> image;
            image.resize(rows, cols);

            in_file.read(reinterpret_cast<char *>(image.data()), rows * cols);

            _images.push_back(image.array().cast<double>() / 255.0);
        }
    } else {
        throw std::runtime_error(path);
    }
}

void MNISTLoader::WriteToDirectory(const std::string &path) {
    throw std::runtime_error("Unimplemented");

    std::filesystem::directory_entry root_dir{path};

    if (!root_dir.exists()) {
        throw std::runtime_error("Cannot open directory for writing MNIST images");
    }

    for (size_t label = 0; label < 10u; label++) {
        std::stringstream ss;
        ss << path << "/" << label;
        std::string this_label_out_dir = ss.str();
        if (!std::filesystem::create_directory(this_label_out_dir)) {
            throw std::runtime_error("Cannot create directory for label");
        }

        for (size_t label_idx = 0; label_idx < _labels.size(); label_idx++) {
            if (static_cast<size_t>(_labels[label_idx]) == label) {
                auto image_data = _images[label_idx];
                std::stringstream out_path_stream;
                out_path_stream << this_label_out_dir << "/" << label_idx << ".bmp";
                std::string out_path = out_path_stream.str();
                // cimg_library::CImg<double> img(image_data.data(), image_data.cols(), image_data.rows(), 1);
                // img.save(out_path.c_str());
            }
        }
    }
}

double MNISTLoader::CalculateMean(std::vector<EigenImage> images) {
    if (!images.size()) {
        return 0.0;
    }

    Eigen::VectorXd means(images.size());
    for (size_t idx = 0; idx < images.size(); idx++) {
        means(idx) = images[idx].mean();
    }

    EigenImage first_image = images[0];
    double image_coeff_count = first_image.cols() * first_image.rows();

    double dataset_mean_numerator = 0.0;
    for (double mean : means) {
        dataset_mean_numerator += image_coeff_count * mean;
    }

    return dataset_mean_numerator / (image_coeff_count * images.size());
}

double MNISTLoader::CalculateStd(std::vector<EigenImage> images) {
    if (!images.size()) {
        return 0.0;
    }

    double total_std_dev_summation = 0.0;
    const double combined_mean = CalculateMean(images);
    EigenImage image_0 = images[0];
    const double image_pixels = image_0.cols() * image_0.rows();

    for (size_t idx = 0; idx < images.size(); idx++) {
        total_std_dev_summation += (images[idx].array() - combined_mean).square().sum();
    }

    double total_std = std::sqrt((1.0 / (image_pixels * images.size())) * total_std_dev_summation);
    return total_std;
}

namespace MNIST {
std::shared_ptr<Vs::Network> Network(size_t pixels_per_image) {
    auto network = std::make_shared<Vs::Network>(pixels_per_image);
    network->AddFullyConnectedLayer(100, Vs::ReLu);
    network->AddFullyConnectedLayer(100, Vs::ReLu);
    network->AddFullyConnectedLayer(10, Vs::ReLu);
    network->AddSoftMaxLayer();

    return network;
}

std::shared_ptr<Vs::Network> MiniNetwork(size_t pixels_per_image) {
    auto network = std::make_shared<Vs::Network>(pixels_per_image);
    network->AddFullyConnectedLayer(10, Vs::ReLu);
    network->AddSoftMaxLayer();

    return network;
}

Vs::IOVector GetOneHotVector(size_t label) {
    Vs::IOVector one_hot(10);
    one_hot.fill(0);
    one_hot(label) = 1;

    return one_hot;
}

Vs::IOVector ImageToNormalizedInput(Vs::EigenImage image, double dataset_mean, double dataset_std) {
    auto column_w_fval = image.reshaped(Eigen::AutoSize, 1).cast<Vs::FVal>();
    return ((column_w_fval.array() - dataset_mean) / dataset_std);
}
}  // namespace MNIST

}  // namespace Vs