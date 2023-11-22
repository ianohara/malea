#ifndef VIEW_SYNTH_HPP
#define VIEW_SYNTH_HPP

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace Vs {

class CameraIntrinsics {
    // TODO(imo)
};

class Camera {
    Eigen::Vector3d position;
    Eigen::Quaterniond direction;

    CameraIntrinsics intrinsics;
};

template <int W, int H>
class View {
    Eigen::Matrix<double, W, H> image;
    Camera capture_camera;
};
};  // namespace Vs
#endif