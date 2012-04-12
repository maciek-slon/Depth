/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>
#include <iostream>
#include <cmath>

#include "NormalEstimator.hpp"
#include "Common/Logger.hpp"

namespace Processors {
namespace NormalEstimator {

NormalEstimator::NormalEstimator(const std::string & name) : Base::Component(name), m_gamma(10000)
{
	LOG(LTRACE) << "Hello NormalEstimator\n";
	for (unsigned int i=0; i<10000; i++) {
		float v = i/2048.0;
		v = std::pow(v, 3)* 6;
		m_gamma[i] = v*6*256;
	}
}

NormalEstimator::~NormalEstimator()
{
	LOG(LTRACE) << "Good bye NormalEstimator\n";
}

bool NormalEstimator::onInit()
{
	LOG(LTRACE) << "NormalEstimator::initialize\n";

	h_onNewImage.setup(this, &NormalEstimator::onNewImage);
	registerHandler("onNewImage", &h_onNewImage);

	registerStream("in_img", &in_img);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	return true;
}

bool NormalEstimator::onFinish()
{
	LOG(LTRACE) << "NormalEstimator::finish\n";

	return true;
}

bool NormalEstimator::onStep()
{
	LOG(LTRACE) << "NormalEstimator::step\n";
	return true;
}

bool NormalEstimator::onStop()
{
	return true;
}

bool NormalEstimator::onStart()
{
	return true;
}

void cross(float a1, float a2, float a3, float b1, float b2, float b3, float & c1, float & c2, float & c3) {
	c1 = a2*b3 - a3*b2;
	c2 = a3*b1 - a1*b3;
	c3 = a1*b2 - a2*b1;
	float nf = 1/sqrt(c1*c1+c2*c2+c3*c3);
	c1 *= nf;
	c2 *= nf;
	c3 *= nf;
}

void NormalEstimator::onNewImage() {
	try {
		img = in_img.read();
		cv::Size size = img.size();
		out.create(size, CV_8UC3);

		//std::cout << size.width << "x" << size.height << "x" << img.channels() << "x" << img.elemSize() << "bpp\n";

		// Check the arrays for continuity and, if this is the case,
		// treat the arrays as 1D vectors
		//if (img.isContinuous()) {
		//	size.width *= size.height;
		//	size.height = 1;
		//}

		//double mmin, mmax;
		//cv::minMaxLoc(img, &mmin, &mmax);
		//std::cout << mmax << "\n";

		//cv::GaussianBlur(img, img, cv::Size(5, 5), 0);

		for (int i = 1; i < size.height-1; i++) {
			float* img_pp = img.ptr <float> (i-1);
			float* img_p =  img.ptr <float> (i);
			float* img_np = img.ptr <float> (i+1);
			uchar* out_p = out.ptr <uchar> (i);

			int j;
			for (j = 1; j < size.width-1; ++j) {
				float x = img_p[j*3];
				float y = img_p[j*3+1];
				float z = img_p[j*3+2];
				float ax = img_pp[j*3] - x;
				float ay = img_pp[j*3+1] - y;
				float az = img_pp[j*3+2] - z;
				float bx = img_p[(j-1)*3] - x;
				float by = img_p[(j-1)*3+1] - y;
				float bz = img_p[(j-1)*3+2] - z;
				float cx = img_np[j*3] - x;
				float cy = img_np[j*3+1] - y;
				float cz = img_np[j*3+2] - z;
				float dx = img_p[(j+1)*3] - x;
				float dy = img_p[(j+1)*3+1] - y;
				float dz = img_p[(j+1)*3+2] - z;

				float rx, ry, rz;
				cross(ax, ay, az, bx, by, bz, rx, ry, rz);
				cross(bx, by, bz, cx, cy, cz, x, y, z);
				rx += x; ry += y; rz += z;
				cross(cx, cy, cz, dx, dy, dz, x, y, z);
				rx += x; ry += y; rz += z;
				cross(dx, dy, dz, ax, ay, az, x, y, z);
				rx += x; ry += y; rz += z;

				float nf = 0.5/sqrt(rx*rx + ry*ry + rz*rz);
				rx *= nf;
				ry *= nf;
				rz *= nf;
				rx += 0.5;
				ry += 0.5;
				rz = 1 - rz + 0.5;

				z = img_p[3*j+2];
				if (z != 0) {
					//out_p[3*j+2] = rx * 255;
					//out_p[3*j+1] = ry * 255;
					//out_p[3*j+0] = rz * 255;
					out_p[3*j+0] = (z-0.5) * 512;
				} else {
					out_p[3*j+2] = 0;
					out_p[3*j+1] = 0;
					out_p[3*j+0] = 0;
				}


				/*if (z != 0) {
					out_p[3*j+2] = (img_p[3*j+0]+1)*128;
					out_p[3*j+1] = (img_p[3*j+1]+1)*128;
					out_p[3*j+0] = img_p[3*j+2] * 100;
				} else {
				}*/

				if (j == (640*240 + 320))
					std::cout << x << ", " << y << ", " << z << "\n";
				/*int val = img_p[j];
				val = (val >> 8) | ( (val & 0xff) << 8);
				//int pval = m_gamma[val]/4;
				int pval = val;
				int lb = pval & 0xff;
				//out_p[3*j] = img_p[j] >> 8;
				if (j == (640*240 + 320))
					std::cout << val << ", " << m_gamma[val]/4 << "\n";
				switch (pval>>8) {
				case 0:
					out_p[3*j+0] = 255;
					out_p[3*j+1] = 255-lb;
					out_p[3*j+2] = 255-lb;
					break;
				case 1:
					out_p[3*j+0] = 255;
					out_p[3*j+1] = lb;
					out_p[3*j+2] = 0;
					break;
				case 2:
					out_p[3*j+0] = 255-lb;
					out_p[3*j+1] = 255;
					out_p[3*j+2] = 0;
					break;
				case 3:
					out_p[3*j+0] = 0;
					out_p[3*j+1] = 255;
					out_p[3*j+2] = lb;
					break;
				case 4:
					out_p[3*j+0] = 0;
					out_p[3*j+1] = 255-lb;
					out_p[3*j+2] = 255;
					break;
				case 5:
					out_p[3*j+0] = 0;
					out_p[3*j+1] = 0;
					out_p[3*j+2] = 255-lb;
					break;
				default:
					out_p[3*j+0] = 0;
					out_p[3*j+1] = 0;
					out_p[3*j+2] = 0;
					break;
				}*/
			}
		}

		cv::medianBlur(out, out, 5);

		out_img.write(out.clone());
		newImage->raise();
	} catch (const exception& ex) {
		LOG(LERROR) << "NormalEstimator::onNewImage() failed. " << ex.what() << endl;
	}
}

}//: namespace NormalEstimator
}//: namespace Processors
