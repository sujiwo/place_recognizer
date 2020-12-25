# place_recognizer: Visual Place Recognition for ROS

## Introduction
This repository provides an implementation of vision-based place recognition for ROS using two approaches: 

- Incremental Bag of Words (IBoW)
- Vector of Locally Aggregated Descriptors (VLAD)

This library only supports Python; however, all critical routines are written in C++. Therefore, performance is maximized, and corresponding libraries and headers for C++ are also provided.

## Requirements

- OpenCV (>= 3.2)
- ROS Melodic (Noetic support is planned)
- Python 2.7 or 3.8 (Python 3 support is currently broken)
- pybind11_catkin
- rosbag_viewer

## SegNet for Segmentation Support (Optional)
Add $CAFFE_SEGNET_DIR/python to your $PYTHONPATH and it will be detected automatically

## Installation

1. Clone and put place_recognizer directory under your ROS Workspace
2. Run `catkin_make install`

## Usage

In order to localize using an image, place_recognizer requires a map file created from image training stream. In addition, VLAD mapping requires visual codebook; this library provides a script to generate visual codebook from Cityscape dataset. 

Mapping process usually takes a long time and not in real-time. Therefore, place_recognizer only supports map creation in offline style that requires accessing ROS bag directly.

### Command Line

For command-line usage, three scripts are provided:

- train_from_bag.py
- cityscape_dict.py
- server (ROS service to recognize and image and returns latitude/longitude; currently broken)

### Python API