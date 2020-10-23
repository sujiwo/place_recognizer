#include <iostream>
#include "conversion.h"
/*
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder.
 */

static void init()
{
    import_array();
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

using namespace cv;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}


class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() { stdAllocator = Mat::getStdAllocator(); }
    ~NumpyAllocator() {}

    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
    {
        UMatData* u = new UMatData(this);
        u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( int i = 0; i < dims - 1; i++ )
            step[i] = (size_t)_strides[i];
        step[dims-1] = CV_ELEM_SIZE(type);
        u->size = sizes[0]*step[0];
        u->userdata = o;
        return u;
    }

    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const
    {
        if( data != 0 )
        {
            CV_Error(Error::StsAssert, "The data should normally be NULL!");
            // probably this is safe to do in such extreme case
            return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
        }
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
        depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
        depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
        depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i, dims = dims0;
        cv::AutoBuffer<npy_intp> _sizes(dims + 1);
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
            _sizes[dims++] = cn;
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        return allocate(o, dims0, sizes, type, step);
    }

    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const
    {
        return stdAllocator->allocate(u, accessFlags, usageFlags);
    }

    void deallocate(UMatData* u) const
    {
        if(!u)
            return;
        PyEnsureGIL gil;
        CV_Assert(u->urefcount >= 0);
        CV_Assert(u->refcount >= 0);
        if(u->refcount == 0)
        {
            PyObject* o = (PyObject*)u->userdata;
            Py_XDECREF(o);
            delete u;
        }
    }

    const MatAllocator* stdAllocator;
};


NumpyAllocator g_numpyAllocator;

NDArrayConverter::NDArrayConverter() { init(); }

void NDArrayConverter::init()
{
    import_array();
}

cv::Mat NDArrayConverter::toMat(const PyObject *o)
{
    cv::Mat m;

    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("toMat: Object is not a numpy array");
    }

    PyArrayObject *oarr = (PyArrayObject*)o;
    int typenum = PyArray_TYPE(oarr);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        failmsg("toMat: Data type = %d is not supported", typenum);
    }

    int ndims = PyArray_NDIM(oarr);

    if(ndims >= CV_MAX_DIM)
    {
        failmsg("toMat: Dimensionality (=%d) is too high", ndims);
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool transposed = false;

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ndims >= 2 && step[0] < step[1] )
    {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }

    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2)
    {
        failmsg("toMat: Object has more than 2 dimensions");
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);

    if( m.data )
    {
//        m.u->refcount = *refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };

    m.allocator = &g_numpyAllocator;

    if( transposed )
    {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
    return m;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->u->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
//    p->addref();
    PyObject *o = (PyObject*)p->u->userdata;
    Py_INCREF(o);
    return o;
}


template<typename _Tp> struct pyopencvVecConverter
{
	static bool to(PyObject* obj, std::vector<_Tp>& value, const ArgInfo info)
	{
		typedef typename DataType<_Tp>::channel_type _Cp;
		if(!obj || obj == Py_None)
			return true;
		if (PyArray_Check(obj))
		{
			Mat m;
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

		int type = DataType<_Tp>::type;
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
				data[0] = saturate_cast<_Cp>(c.real);
				data[1] = saturate_cast<_Cp>(c.imag);
				continue;
			}
			if( channels > 1 )
			{
				if( PyArray_Check(item))
				{
					Mat src;
					pyopencv_to(item, src, info);
					if( src.dims != 2 || src.channels() != 1 ||
							((src.cols != 1 || src.rows != channels) &&
									(src.cols != channels || src.rows != 1)))
						break;
					Mat dst(src.rows, src.cols, depth, data);
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
					data[j] = saturate_cast<_Cp>(v);
				}
				else if( PyLong_Check(item_ij))
				{
					int v = (int)PyLong_AsLong(item_ij);
					if( v == -1 && PyErr_Occurred() )
						break;
					data[j] = saturate_cast<_Cp>(v);
				}
				else if( PyFloat_Check(item_ij))
				{
					double v = PyFloat_AsDouble(item_ij);
					if( PyErr_Occurred() )
						break;
					data[j] = saturate_cast<_Cp>(v);
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
		Mat src((int)value.size(), DataType<_Tp>::channels, DataType<_Tp>::depth, (uchar*)&value[0]);
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

template<> struct pyopencvVecConverter<KeyPoint>
{
    static bool to(PyObject* obj, std::vector<KeyPoint>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const std::vector<KeyPoint>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};


