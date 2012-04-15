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
#include "Common/Timer.hpp"

namespace Processors {
namespace NormalEstimator {

NormalEstimator::NormalEstimator(const std::string & name) : Base::Component(name),
		prop_radius("radius", 0.01)
{
	LOG(LTRACE) << "Hello NormalEstimator\n";
	registerProperty(prop_radius);
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

	newNormals = registerEvent("newNormals");

	registerStream("out_normals", &out_normals);

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

float InvSqrt(float x){
   float xhalf = 0.5f * x;
   int i = *(int*)&x; // store floating-point bits in integer
   i = 0x5f3759d5 - (i >> 1); // initial guess for Newton's method
   x = *(float*)&i; // convert new bits into float
   x = x*(1.5f - xhalf*x*x); // One round of Newton's method
   return x;
}

float calculateDist(cv::Point3f a, cv::Point3f b) {
	return norm(a-b);
}

cv::Point3f calculateCross(cv::Point3f a, cv::Point3f b) {
	cv::Point3f c;
	c.x = a.y*b.z - a.z*b.y;
	c.y = a.z*b.x - a.x*b.z;
	c.z = a.x*b.y - a.y*b.x;
	return c;
}

cv::Point3f calculateNormal(cv::Mat img, cv::Mat der_row, cv::Mat der_col, int row, int col, float dist, int window) {
	cv::Point3f ret;
	cv::Point3f curpoint = img.at<cv::Point3f>(row, col);
	cv::Point3f drow(0, 0, 0), dcol(0, 0, 0);
	cv::Point3f pt;
	float sum=0;

	for (int i = -window; i <= window; ++i) {
		cv::Point3f * drow_ptr = der_row.ptr<cv::Point3f>(row+i);
		cv::Point3f * dcol_ptr = der_col.ptr<cv::Point3f>(row+i);
		cv::Point3f * img_ptr = img.ptr<cv::Point3f>(row+i);
		for (int j = -window; j <= window; ++j) {
			pt = img_ptr[col+j];
			cv::Point3f tmp = curpoint-pt;
			//float d = calculateDist(curpoint, pt);
			float dd = InvSqrt(tmp.x*tmp.x+tmp.y*tmp.y+tmp.z*tmp.z);
			float d = 1/dd;
			if (d <= dist) {
				float sc = 1.0 - d/dist;
				drow += drow_ptr[col+j]*sc;
				dcol += dcol_ptr[col+j]*sc;
				//sum += sc;
			}
		}
	}
//	drow *= (1./sum);
//	dcol *= (1./sum);


	ret = calculateCross(drow, dcol);
	if (ret.z < 0)
		ret = -ret;
	//if (norm(ret) > 0.001)
	//	ret = cv::Point3f(-1, -1, -1);
	//else
		ret *= (1./norm(ret));

	return ret;
}

void NormalEstimator::onNewImage() {
	try {
		Common::Timer timer;
		timer.restart();
		img = in_img.read();
		cv::Size size = img.size();
		out.create(size, CV_8UC3);
		cv::Mat der_row;
		cv::Mat der_col;
		der_row.create(size, CV_32FC3);
		der_col.create(size, CV_32FC3);
		normals.create(size, CV_32FC3);

		float t1, t2, t3, t4;

		timer.restart();
		for (int i = 0; i < size.height-1; i++) {
			cv::Point3f* img_p = img.ptr <cv::Point3f> (i);
			cv::Point3f* img_np =  img.ptr <cv::Point3f> (i+1);
			cv::Point3f* p_row = der_row.ptr<cv::Point3f>(i);
			cv::Point3f* p_col = der_col.ptr<cv::Point3f>(i);
			for (int j = 0; j < size.width-1; ++j) {
				p_row[j] = img_p[j+1] - img_p[j];
				if (fabs(p_row[j].z) > 0.05) p_row[j]=cv::Point3f(0, 0, 0);
				p_col[j] = img_np[j] - img_p[j];
				if (fabs(p_col[j].z) > 0.05) p_col[j]=cv::Point3f(0, 0, 0);
			}
		}
		t1 = timer.elapsed();

		int window = 5;
		for (int i = window; i < size.height-window-1; i++) {
			uchar * out_p = out.ptr<uchar>(i);
			cv::Point3f * nptr = normals.ptr<cv::Point3f>(i);
			for (int j = window; j < size.width-window-1; ++j) {
				cv::Point3f normal = calculateNormal(img, der_row, der_col, i, j, prop_radius, window);
				out_p[3*j+2] = 0.5*(normal.x+1) * 255;
				out_p[3*j+1] = 0.5*(normal.y+1) * 255;
				out_p[3*j+0] = 0.5*(normal.z+1) * 255;
				//out_p[3*j+0] = der_row.at<cv::Point3f>(i, j).x * 400;
				nptr[j] = normal;
			}
		}
		t2 = timer.elapsed();

		LOG(LNOTICE) << t1 << ", " << t2-t1;
		out_img.write(out.clone());
		newImage->raise();
		out_normals.write(normals);
		newNormals->raise();
	} catch (const exception& ex) {
		LOG(LERROR) << "NormalEstimator::onNewImage() failed. " << ex.what() << endl;
	}
}

}//: namespace NormalEstimator
}//: namespace Processors
