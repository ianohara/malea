#include <string>
#include <vector>

#include "Eigen/Core"

namespace Vs {
    typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> EigenImage;

    class MNISTLoader {
    public:
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
}