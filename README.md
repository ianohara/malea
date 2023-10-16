# View Synthesis Tools
A repository that implements various view synthesis algorithms (eg: NeRF)

## Architecture Notes

This implements the original [Neural Radiance Fields (NeRF) paper](https://doi.org/10.48550/arXiv.2003.08934) (hereafter, the NeRF paper).

The ML goal is to learn a function of f(x, y, z, theta, phi) -> r,g,b,sigma where sigma is a "volume density".  The model of used is found on pg 18 of the NeRF paper, and is a mostly just a bunch of fully connected layers with inputs concatenated at various layers.  See pg 18 for a good diagram and description.

The parts of a full implementation include:
  * The network
  * The ability to evaluate the network given weights and an input vector
  * The ability to take a requested output image(s), and fill in each pixel by evaluating the network
  * The ability to load training images, and corresponding camera locations and properties
  * The ability to use the training images and camera information to train the network used to evaluate new output images of a particular scene

This naturally breaks into 2 different stages:
  1. Training
  2. Output Image Generation

### Training

Training assumes that:
  1. Images of a single static scene are available ("Input Images")
  2. For each Input Image, the pose and camera properties (extrinsics and intrinsics in computer vision parlance, "Camera Infos") are available


### Output Image Generation

Output Image Generation assumes that:
  1. Trained network weights are available for the scene associated with the output images
  2. The post and camera properties of the camera(s) for the output images are available



