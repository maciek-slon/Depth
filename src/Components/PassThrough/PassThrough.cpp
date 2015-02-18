/*!
 * \file
 * \brief
 * \author Maciej Stefanczyk
 */

#include <memory>
#include <string>

#include "PassThrough.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

namespace Processors {
namespace PassThrough {

PassThrough::PassThrough(const std::string & name) :
		Base::Component(name) , 
		z_min("z_min", 0), 
		z_max("z_max", 10) {
	registerProperty(z_min);
	registerProperty(z_max);

}

PassThrough::~PassThrough() {
}

void PassThrough::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("in_xyz", &in_xyz);
	registerStream("out_xyz", &out_xyz);
	registerStream("out_mask", &out_mask);
	// Register handlers
	registerHandler("onNewImage", boost::bind(&PassThrough::onNewImage, this));
	addDependency("onNewImage", &in_xyz);

}

bool PassThrough::onInit() {

	return true;
}

bool PassThrough::onFinish() {
	return true;
}

bool PassThrough::onStop() {
	return true;
}

bool PassThrough::onStart() {
	return true;
}

void PassThrough::onNewImage() {
	cv::Mat img = in_xyz.read().clone();
	cv::Mat mask = cv::Mat::zeros(img.size(), CV_8UC1);
	
	int rows = img.rows;
	int cols = img.cols;
	
	if (img.isContinuous() && mask.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	
	int i,j;
	float* p;
	uchar* mp;
		
	for( i = 0; i < rows; ++i) {
		p = img.ptr<float>(i);
		mp = mask.ptr<uchar>(i);
		for ( j = 0; j < cols; ++j) {
			// read point coordinates 
			float x = p[3*j];
			float y = p[3*j + 1];
			float z = p[3*j + 2];
			
			if ( (z < z_min) || (z > z_max) ) {
				p[3*j] = 0;
				p[3*j+1] = 0;
				p[3*j+2] = 0;
			} else if (std::isfinite(z)){
				mp[j] = 255;
			}
		}
	}
	
	out_xyz.write(img);
	out_mask.write(mask);
}



} //: namespace PassThrough
} //: namespace Processors
