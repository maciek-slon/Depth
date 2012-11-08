/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>

#include "DepthMapGenerator.hpp"
#include "Common/Logger.hpp"

#include <cmath>

namespace Processors {
namespace DepthMapGenerator {

DepthMapGenerator::DepthMapGenerator(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello DepthMapGenerator\n";
	m_width = 640;
	m_height = 480;
}

DepthMapGenerator::~DepthMapGenerator()
{
	LOG(LTRACE) << "Good bye DepthMapGenerator\n";
}

void DepthMapGenerator::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);
}

bool DepthMapGenerator::onInit()
{
	LOG(LTRACE) << "DepthMapGenerator::initialize\n";

	return true;
}

bool DepthMapGenerator::onFinish()
{
	LOG(LTRACE) << "DepthMapGenerator::finish\n";

	return true;
}

bool DepthMapGenerator::onStep()
{
	LOG(LTRACE) << "DepthMapGenerator::step\n";
	m_image.create(cv::Size(m_width, m_height), CV_32FC3);

	cv::RNG rng;
	cv::Point3f pt;
	int cx = 320, cy = 240;
	float r = 150;
	float sx = 0.001;
	float sy = 0.001;
	float sz = 0.001;
	float noise;
	for (int i = 0; i < m_height; i++) {
		cv::Point3f* img_p = m_image.ptr <cv::Point3f> (i);
		for (int j = 0; j < m_width; ++j) {
			float dx = cx - j;
			float dy = cy - i;
			float d = sqrt(dx*dx+dy*dy);
			noise = rng.gaussian(0.001);
			if (d > r)
				pt = cv::Point3f((j-m_width/2)*sx, (i-m_height/2)*sy, 1.1 + noise);
			else
				pt = cv::Point3f((j-m_width/2)*sx, (i-m_height/2)*sy, 1-sqrt(r*r-d*d)*sz+noise);
			img_p[j] = pt;
		}
	}

	out_img.write(m_image);
	newImage->raise();

	return true;
}

bool DepthMapGenerator::onStop()
{
	return true;
}

bool DepthMapGenerator::onStart()
{
	return true;
}

}//: namespace DepthMapGenerator
}//: namespace Processors
