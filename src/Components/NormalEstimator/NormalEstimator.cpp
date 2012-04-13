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

float calculateDist(cv::Point3f a, cv::Point3f b) {
	return norm(a-b);
}

cv::Point3f calculateCross(cv::Point3f a, cv::Point3f b) {
	cv::Point3f c;
	c.x = a.y*b.z - a.z*b.y;
	c.y = a.z*b.x - a.x*b.z;
	c.z = a.x*b.y - a.y*b.x;
	return c * (1./norm(c));
}

cv::Point3f calculateNormal(cv::Mat img, cv::Mat der_row, cv::Mat der_col, int row, int col, float dist, int window) {
	cv::Point3f ret;
	cv::Point3f curpoint = img.at<cv::Point3f>(row, col);
	cv::Point3f drow(0, 0, 0), dcol(0, 0, 0);
	cv::Point3f pt;
	float sum=0;

	for (int i = -window; i <= window; ++i) {
		for (int j = -window; j <= window; ++j) {
			pt = img.at<cv::Point3f>(row+i, col+j);
			float d = calculateDist(curpoint, pt);
			if (d <= dist) {
				float sc = 1.0 - d/dist;
				drow += der_row.at<cv::Point3f>(row+i, col+j) * sc;
				dcol += der_col.at<cv::Point3f>(row+i, col+j) * sc;
				sum += sc;
			}
		}
	}
//	drow *= (1./sum);
//	dcol *= (1./sum);

	ret = calculateCross(drow, dcol);
	if (row == 240 && col == 320)
		std::cout << ret.x << ", " << ret.y << ", " << ret.z << " | " << norm(ret) << "\n";
	return ret;
}

void NormalEstimator::onNewImage() {
	try {
		img = in_img.read();
		cv::Size size = img.size();
		out.create(size, CV_8UC3);
		cv::Mat der_row;
		cv::Mat der_col;
		der_row.create(size, CV_32FC3);
		der_col.create(size, CV_32FC3);

		for (int i = 0; i < size.height-1; i++) {
			cv::Point3f* img_p = img.ptr <cv::Point3f> (i);
			cv::Point3f* img_np =  img.ptr <cv::Point3f> (i+1);
			cv::Point3f* p_row = der_row.ptr<cv::Point3f>(i);
			cv::Point3f* p_col = der_col.ptr<cv::Point3f>(i);
			for (int j = 0; j < size.width-1; ++j) {
				p_row[j] = img_p[j+1] - img_p[j];
				p_col[j] = img_np[j] - img_p[j];
			}
		}

		int window = 5;
		for (int i = window; i < size.height-window-1; i++) {
			uchar * out_p = out.ptr<uchar>(i);
			for (int j = window; j < size.width-window-1; ++j) {
				cv::Point3f normal = calculateNormal(img, der_row, der_col, i, j, 0.02, window);
				out_p[3*j+2] = normal.x * 255;
				out_p[3*j+1] = normal.y * 255;
				out_p[3*j+0] = normal.z * 255;
			}
		}

		/*for (int i = 1; i < size.height-1; i++) {
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
					out_p[3*j+2] = rx * 255;
					out_p[3*j+1] = ry * 255;
					out_p[3*j+0] = rz * 255;
					//out_p[3*j+0] = (z-0.5) * 512;
				} else {
					out_p[3*j+2] = 0;
					out_p[3*j+1] = 0;
					out_p[3*j+0] = 0;
				}
			}
		}*/

		out_img.write(out.clone());
		newImage->raise();
	} catch (const exception& ex) {
		LOG(LERROR) << "NormalEstimator::onNewImage() failed. " << ex.what() << endl;
	}
}

}//: namespace NormalEstimator
}//: namespace Processors
