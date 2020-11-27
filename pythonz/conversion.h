# ifndef __CONVERSION_OPENCV_H__
# define __CONVERSION_OPENCV_H__

#include <vector>
#include <Python.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

static PyObject* opencv_error = 0;

typedef std::vector<uchar> vector_uchar;
typedef std::vector<char> vector_char;
typedef std::vector<int> vector_int;
typedef std::vector<float> vector_float;
typedef std::vector<double> vector_double;
typedef std::vector<cv::Point> vector_Point;
typedef std::vector<cv::Point2f> vector_Point2f;
typedef std::vector<cv::Point3f> vector_Point3f;
typedef std::vector<cv::Vec2f> vector_Vec2f;
typedef std::vector<cv::Vec3f> vector_Vec3f;
typedef std::vector<cv::Vec4f> vector_Vec4f;
typedef std::vector<cv::Vec6f> vector_Vec6f;
typedef std::vector<cv::Vec4i> vector_Vec4i;
typedef std::vector<cv::Rect> vector_Rect;
typedef std::vector<cv::Rect2d> vector_Rect2d;
typedef std::vector<cv::KeyPoint> vector_KeyPoint;
typedef std::vector<cv::Mat> vector_Mat;
typedef std::vector<cv::UMat> vector_UMat;
typedef std::vector<cv::DMatch> vector_DMatch;
typedef std::vector<cv::String> vector_String;
typedef std::vector<cv::Scalar> vector_Scalar;

typedef std::vector<std::vector<char> > vector_vector_char;
typedef std::vector<std::vector<cv::Point> > vector_vector_Point;
typedef std::vector<std::vector<cv::Point2f> > vector_vector_Point2f;
typedef std::vector<std::vector<cv::Point3f> > vector_vector_Point3f;
typedef std::vector<std::vector<cv::DMatch> > vector_vector_DMatch;
typedef std::vector<std::vector<cv::KeyPoint> > vector_vector_KeyPoint;



template<typename T> static
bool pyopencv_to(PyObject* obj, T& p, const char* name = "<unknown>");

template<typename T> static
PyObject* pyopencv_from(const T& src);

template<> bool pyopencv_to (PyObject* o, cv::Mat& m, const char* name);
template<> PyObject* pyopencv_from (const cv::Mat& m);

template<> bool pyopencv_to (PyObject* obj, cv::KeyPoint& kp, const char* name);
template<> PyObject* pyopencv_from<cv::KeyPoint> (const cv::KeyPoint& m);

# endif
