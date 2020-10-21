/*
 * binding.cpp
 *
 *  Created on: Oct 21, 2020
 *      Author: sujiwo
 */

#include <iostream>
#include <Python.h>
#include "IncrementalBoW.h"


static PyMethodDef place_recognizer_Methods[] = {
/*
    {"im_test", method_im_test, METH_NOARGS, "Test Method"},
	{"autoAdjustGammaRGB", method_autoAdjustGammaRGB, METH_VARARGS, "Automatic gamma adjusment"},
	{"multiScaleRetinexCP", method_multiScaleRetinexCP, METH_VARARGS, "Multi-scale Retinex with Color Preservation"},
	{"dynamicHistogramEqualization", method_dynamicHistogramEqualization, METH_VARARGS, "Dynamic Histogram Equalization"},
	{"exposureFusion", method_exposureFusion, METH_VARARGS, "Exposure Fusion"},
*/
    {NULL, NULL, 0, NULL}
};


PyMODINIT_FUNC
init_place_recognizer(void)
{
	(void) Py_InitModule("im_enhance", place_recognizer_Methods);
}
