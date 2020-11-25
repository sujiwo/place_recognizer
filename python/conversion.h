# ifndef __CONVERSION_OPENCV_H__
# define __CONVERSION_OPENCV_H__

#include <vector>
#include <Python.h>
#
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...);

class PyAllowThreads;

class PyEnsureGIL;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

static PyObject* failmsgp(const char *fmt, ...);

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}


class NumpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

class NDArrayConverter
{
private:
    void init();
public:
    NDArrayConverter();
    cv::Mat toMat(const PyObject* o);
    PyObject* toNDArray(const cv::Mat& mat);
};


struct ArgInfo
{
	const char * name;
	bool outputarg;
	// more fields may be added if necessary

	ArgInfo(const char * name_, bool outputarg_)
	: name(name_)
	, outputarg(outputarg_) {}

	// to match with older pyopencv_to function signature
	operator const char *() const { return name; }
};

template<typename T> static
bool pyopencv_to(PyObject* obj, T& p, const char* name = "<unknown>");

template<typename T> static
PyObject* pyopencv_from(const T& src);


inline bool objectToVectorKeyPoint(PyObject* obj, std::vector<cv::KeyPoint> &vec)
{
    if (!PySequence_Check(obj))
        return false;
    PyObject *seq = PySequence_Fast(obj, "KeyPoint");
    if (seq == NULL)
        return false;
    int i, n = (int)PySequence_Fast_GET_SIZE(seq);
    vec.resize(n);

    PyObject** items = PySequence_Fast_ITEMS(seq);
    for (i=0; i<n; ++i) {
    	PyObject *item = items[i];
    	PyArg_ParseTuple(item, "(dd)fffii", vec[i].pt.x, vec[i].pt.y, vec[i].size, vec[i].angle, vec[i].response, vec[i].octave, vec[i].class_id);
    }

    Py_DECREF(seq);
    return i==n;
}

# endif
