/*
 * binding.cpp
 *
 *  Created on: Oct 21, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <Python.h>
#include <structmember.h>
#include "BOWKmajorityTrainer.h"
#include "IncrementalBoW.h"
#include "conversion.h"

using namespace std;


/*
 * Documentation: https://docs.python.org/2.7/extending/newtypes.html
 */

struct Place_Recognizer_iBoWDb {
	PyObject_HEAD
	PlaceRecognizer::IncrementalBoW imageDb;
} ;

static void iBoWDB_dealloc (Place_Recognizer_iBoWDb *self)
{
	Py_TYPE(self)->tp_free((PyObject*)self);
}

/*
 * Object allocation
 */
static PyObject* iBoWDB_new (PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	Place_Recognizer_iBoWDb *self = (Place_Recognizer_iBoWDb*)type->tp_alloc(type, 0);
	return (PyObject*)self;
}

/*
 * Object initialization
 */
static void iBoWDB_init(Place_Recognizer_iBoWDb *self, PyObject *args, PyObject *kwds)
{

}

static void iBoWDB_addImage(Place_Recognizer_iBoWDb *self, PyObject *args)
{

}

static void iBoWDB_addImage2(Place_Recognizer_iBoWDb *self, PyObject *args)
{

}

static PyObject* iBoWDB_search(Place_Recognizer_iBoWDb *self, PyObject *args)
{

}


static PyObject* iBoWDB_save(Place_Recognizer_iBoWDb *self, PyObject *args)
{

}

static PyObject* iBoWDB_load(Place_Recognizer_iBoWDb *self, PyObject *args)
{

}

static PyObject *py_kmajority(PyObject *self, PyObject *args)
{
	NDArrayConverter cvt;
	PyObject *img_o;
	int num_clusters;
	PyArg_ParseTuple(args, "Oi", &img_o, &num_clusters);

	cout << "Clustering with " << num_clusters << endl;

	cv::Mat descriptors_in, centers;
	descriptors_in = cvt.toMat(img_o);

	cout << "Descriptor sizes:" << descriptors_in.rows << 'x' << descriptors_in.cols << endl;

	cv::BOWKmajorityTrainer kmj(num_clusters);
	centers = kmj.cluster(descriptors_in);

	PyObject *obj_np = cvt.toNDArray(centers);
	return obj_np;
}

static PyMethodDef iBoWDB_methods[] = {
	{"addImage", (PyCFunction)iBoWDB_addImage, METH_VARARGS, "Add first image to database"},
	{"addImage2", (PyCFunction)iBoWDB_addImage, METH_VARARGS, "Add subsequent image to database"},
	{"search", (PyCFunction)iBoWDB_search, METH_VARARGS, "Search database using list of descriptors"},
	{"save", (PyCFunction)iBoWDB_save, METH_VARARGS, "Save map to file"},
	{"load", (PyCFunction)iBoWDB_load, METH_VARARGS, "Load map from file"},
	{NULL}  /* Sentinel */
};

static PyTypeObject iBoWDB_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "place_recognizer.iBoW",             /* tp_name */
    sizeof(Place_Recognizer_iBoWDb),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)iBoWDB_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "Incremental Bag-of-Words",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
	iBoWDB_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)iBoWDB_init,      /* tp_init */
    0,                         /* tp_alloc */
    iBoWDB_new                 /* tp_new */
};

static PyMethodDef place_recognizer_Methods[] = {
	{"kmajority", py_kmajority, METH_VARARGS, "Cluster binary features input with K-Majority Algorithm"},
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
init_place_recognizer(void)
{
	PyObject* m;
	if (PyType_Ready(&iBoWDB_Type) < 0)
		return;

	m = Py_InitModule3("_place_recognizer", place_recognizer_Methods, "Place recognition based on image");
	if (m==NULL)
		return;

	Py_INCREF(&iBoWDB_Type);
	PyModule_AddObject(m, "iBoW", (PyObject*)&iBoWDB_Type);
}
