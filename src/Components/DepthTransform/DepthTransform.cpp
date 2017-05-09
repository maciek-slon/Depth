/*!
 * \file
 * \brief
 * \author Łukasz Żmuda
 */

#include <memory>
#include <string>

#include "DepthTransform.hpp"
#include "Common/Logger.hpp"
#include <Types/MatrixTranslator.hpp>

#include <boost/bind.hpp>
#include <boost/format.hpp>

namespace Processors {
namespace DepthTransform {
  
using Types::HomogMatrix;

DepthTransform::DepthTransform(const std::string & name) :
	Base::Component(name),
	prop_inverse("inverse", false),
	pass_through("pass_through", false)
{
	registerProperty(prop_inverse);
	registerProperty(pass_through);
}

DepthTransform::~DepthTransform() {

}

void DepthTransform::prepareInterface() {

	// Register data streams, events and event handlers HERE!
	registerStream("in_homogMatrix", &in_homogMatrix);
	registerStream("in_depth_xyz", &in_image_xyz);
	registerStream("out_depth_xyz", &out_image_xyz);

	// Register handlers
	registerHandler("DepthTransformation", boost::bind(&DepthTransform::DepthTransformation, this));
	addDependency("DepthTransformation", &in_image_xyz);
	addDependency("DepthTransformation", &in_homogMatrix);

}

bool DepthTransform::onInit() {

	return true;
}

bool DepthTransform::onFinish() {
	return true;
}

bool DepthTransform::onStop() {
	return true;
}

bool DepthTransform::onStart() {
	return true;
}

void DepthTransform::DepthTransformation() {
	try{
	  
	cv::Mat img = in_image_xyz.read();
	cv::Mat out_img;

	HomogMatrix tmp_hm = in_homogMatrix.read();
	HomogMatrix hm;

	CLOG(LDEBUG) << "Input homogenous matrix:\n" << tmp_hm;

	// Check inversion property.
	if (prop_inverse)
		hm.matrix() = tmp_hm.matrix().inverse();
	else
		hm = tmp_hm;
	CLOG(LINFO) << "Using Homogenous matrix (after inversion):\n" << hm;

	// If passthrough - return the input image.
	if (pass_through) {
		CLOG(LINFO) << "Passthough mode on - returning original image";
		out_image_xyz.write(img);
		return;
	}//: if

	// check, if image has proper number of channels
	if (img.channels() != 3) {
		CLOG(LERROR) << "Wrong number of channels";
		return;
	}//: if
	
	// check image depth, allowed is only 32F and 64F
	int img_type = img.depth();
	if ( (img_type != CV_32F) && (img_type != CV_64F) ) {
		CLOG(LERROR) << "Wrong depth";
		return;
	}
	

	// Perform transformation of coordinates.
	cv::Matx44d H = hm;
	perspectiveTransform(img, out_img, H);


	// Check size.	
	int rows = out_img.rows;
	int cols = out_img.cols;
	int i,j;

	if (img_type == CV_32F) {
		// float variant
		float* p;
		for( i = 0; i < rows; ++i) {
			p = out_img.ptr<float>(i);
			for ( j = 0; j < cols; ++j) {
				// Filter invalid numbers.
				if ( (fabs(p[3*j ]) > MAX_RANGE) ||  (fabs(p[3*j + 1]) > MAX_RANGE) || (fabs(p[3*j + 2]) > MAX_RANGE) )
					p[3*j] = p[3*j+1] = p[3*j+2] = INVALID_COORDINATE;
			}//: for
		}//: for
	} else {
		// double variant
		double* p;
		for( i = 0; i < rows; ++i) {
			p = out_img.ptr<double>(i);
			for ( j = 0; j < cols; ++j) {
				// Filter invalid numbers.
				if ( (fabs(p[3*j ]) > MAX_RANGE) ||  (fabs(p[3*j + 1]) > MAX_RANGE) || (fabs(p[3*j + 2]) > MAX_RANGE) )
					p[3*j] = p[3*j+1] = p[3*j+2] = INVALID_COORDINATE;
			}//: for
		}//: for
	}//: if

	// Return image.
	out_image_xyz.write(out_img);

	} catch (...)
	{
		LOG(LERROR) << "Error occured in processing input";
	}
}


} //: namespace DepthTransform
} //: namespace Processors
