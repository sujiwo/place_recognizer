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


template<typename _Tp> struct pyopencvVecConverter
{
	static bool to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
	{
		typedef typename cv::DataType<_Tp>::channel_type _Cp;
		if(!obj || obj == Py_None)
			return true;
		if (PyArray_Check(obj))
		{
			cv::Mat m;
			pyopencv_to(obj, m, info);
			m.copyTo(value);
		}
		if (!PySequence_Check(obj))
			return false;
		PyObject *seq = PySequence_Fast(obj, info.name);
		if (seq == NULL)
			return false;
		int i, j, n = (int)PySequence_Fast_GET_SIZE(seq);
		value.resize(n);

		int type = cv::DataType<_Tp>::type;
		int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
		PyObject** items = PySequence_Fast_ITEMS(seq);

		for( i = 0; i < n; i++ )
		{
			PyObject* item = items[i];
			PyObject* seq_i = 0;
			PyObject** items_i = &item;
			_Cp* data = (_Cp*)&value[i];

			if( channels == 2 && PyComplex_CheckExact(item) )
			{
				Py_complex c = PyComplex_AsCComplex(obj);
				data[0] = cv::saturate_cast<_Cp>(c.real);
				data[1] = cv::saturate_cast<_Cp>(c.imag);
				continue;
			}
			if( channels > 1 )
			{
				if( PyArray_Check(item))
				{
					cv::Mat src;
					pyopencv_to(item, src, info);
					if( src.dims != 2 || src.channels() != 1 ||
							((src.cols != 1 || src.rows != channels) &&
									(src.cols != channels || src.rows != 1)))
						break;
					cv::Mat dst(src.rows, src.cols, depth, data);
					src.convertTo(dst, type);
					if( dst.data != (uchar*)data )
						break;
					continue;
				}

				seq_i = PySequence_Fast(item, info.name);
				if( !seq_i || (int)PySequence_Fast_GET_SIZE(seq_i) != channels )
				{
					Py_XDECREF(seq_i);
					break;
				}
				items_i = PySequence_Fast_ITEMS(seq_i);
			}

			for( j = 0; j < channels; j++ )
			{
				PyObject* item_ij = items_i[j];
				if( PyInt_Check(item_ij))
				{
					int v = (int)PyInt_AsLong(item_ij);
					if( v == -1 && PyErr_Occurred() )
						break;
					data[j] = cv::saturate_cast<_Cp>(v);
				}
				else if( PyLong_Check(item_ij))
				{
					int v = (int)PyLong_AsLong(item_ij);
					if( v == -1 && PyErr_Occurred() )
						break;
					data[j] = cv::saturate_cast<_Cp>(v);
				}
				else if( PyFloat_Check(item_ij))
				{
					double v = PyFloat_AsDouble(item_ij);
					if( PyErr_Occurred() )
						break;
					data[j] = cv::saturate_cast<_Cp>(v);
				}
				else
					break;
			}
			Py_XDECREF(seq_i);
			if( j < channels )
				break;
		}
		Py_DECREF(seq);
		return i == n;
	}

	static PyObject* from(const std::vector<_Tp>& value)
	{
		if(value.empty())
			return PyTuple_New(0);
		cv::Mat src((int)value.size(), cv::DataType<_Tp>::channels, cv::DataType<_Tp>::depth, (uchar*)&value[0]);
		return pyopencv_from(src);
	}
};


template<typename _Tp>
bool pyopencv_to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
{
    return pyopencvVecConverter<_Tp>::to(obj, value, info);
}

template<typename _Tp>
PyObject* pyopencv_from(const std::vector<_Tp>& value)
{
    return pyopencvVecConverter<_Tp>::from(value);
}


template<typename _Tp> static inline bool pyopencv_to_generic_vec(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
{
    if(!obj || obj == Py_None)
       return true;
    if (!PySequence_Check(obj))
        return false;
    PyObject *seq = PySequence_Fast(obj, info.name);
    if (seq == NULL)
        return false;
    int i, n = (int)PySequence_Fast_GET_SIZE(seq);
    value.resize(n);

    PyObject** items = PySequence_Fast_ITEMS(seq);

    for( i = 0; i < n; i++ )
    {
        PyObject* item = items[i];
        if(!pyopencv_to(item, value[i], info))
            break;
    }
    Py_DECREF(seq);
    return i == n;
}

template<typename _Tp> static inline PyObject* pyopencv_from_generic_vec(const std::vector<_Tp>& value)
{
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        PyObject* item = pyopencv_from(value[i]);
        if(!item)
            break;
        PyList_SET_ITEM(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}

template<> struct pyopencvVecConverter<cv::KeyPoint>;

template<>
bool pyopencv_to(PyObject* obj, cv::KeyPoint& kp, const char* name);
template<>
PyObject* pyopencv_from(const cv::KeyPoint& value);


# endif
