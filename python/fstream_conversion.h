/*
 * fstream_conversion.h
 *
 *  Created on: Jan 9, 2021
 *      Author: sujiwo
 */

#ifndef PLACE_RECOGNIZER_FSTREAM_CONVERSION_H_
#define PLACE_RECOGNIZER_FSTREAM_CONVERSION_H_

#include <string>
#include <fstream>
#include <ext/stdio_filebuf.h>
#include <ext/stdio_sync_filebuf.h>
#include <pybind11/cast.h>


namespace py = pybind11;

typedef __gnu_cxx::stdio_filebuf<char> BinaryStream;


namespace pybind11 { namespace detail {


template<>
struct type_caster<BinaryStream> {
public:
	PYBIND11_TYPE_CASTER(BinaryStream, _("file"));

	//! 1. cast file object to fstream
	bool load(handle src, bool)
	{
		obj = py::reinterpret_borrow<object>(src);

		// Set modes
		std::string modes = py::str(getattr(src, "mode"));
		std::ios_base::openmode m;
		if (modes.find("b")!=std::string::npos)
			m |= std::fstream::binary;
		if (modes.find("r")!=std::string::npos)
			m |= std::fstream::in;
		if (modes.find("w")!=std::string::npos)
			m |= std::fstream::out;

		// XXX: Check compatibility with Python3
		value = BinaryStream(PyFile_AsFile(obj.ptr()), m);
		return true;
	}

	//! 2. cast fstream to file object
	static handle cast(const std::fstream& mat, return_value_policy, handle defval);

private:
	object obj;
	std::shared_ptr<std::fstream> v;
};

}}

#endif /* PLACE_RECOGNIZER_FSTREAM_CONVERSION_H_ */
