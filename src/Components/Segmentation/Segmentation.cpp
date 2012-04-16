/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>
#include <queue>

#include "Segmentation.hpp"
#include "Common/Logger.hpp"

namespace Processors {
namespace Segmentation {

Segmentation::Segmentation(const std::string & name) :
		Base::Component(name) {
	LOG(LTRACE) << "Hello Segmentation\n";
	m_normals_ready = m_depth_ready = false;
}

Segmentation::~Segmentation() {
	LOG(LTRACE) << "Good bye Segmentation\n";
}

bool Segmentation::onInit() {
	LOG(LTRACE) << "Segmentation::initialize\n";

	// Register data streams, events and event handlers HERE!

	h_onNewDepth.setup(this, &Segmentation::onNewDepth);
	registerHandler("onNewDepth", &h_onNewDepth);

	h_onNewNormals.setup(this, &Segmentation::onNewNormals);
	registerHandler("onNewNormals", &h_onNewNormals);

	registerStream("in_depth", &in_depth);
	registerStream("in_normals", &in_normals);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	return true;
}

bool Segmentation::onFinish() {
	LOG(LTRACE) << "Segmentation::finish\n";

	return true;
}

bool Segmentation::check(cv::Point point, cv::Point dir) {
	cv::Point dest = point+dir;
	if (m_closed.at<uchar>(dest) == 255)
		return false;

	m_closed.at<uchar>(dest) = 255;

	if (!dest.inside(cv::Rect(0, 0, 639, 479)))
		return false;

	cv::Point3f curn = m_normals.at<cv::Point3f>(point);
	cv::Point3f curd = m_depth.at<cv::Point3f>(point);
	cv::Point3f desn = m_normals.at<cv::Point3f>(point+dir);
	cv::Point3f desd = m_depth.at<cv::Point3f>(point+dir);

	float angle = 180. / 3.14 * acos( curn.dot(desn) );
	float dist = norm(desd-curd);
	return (angle < 4 && dist < 0.02);
}

bool Segmentation::onStep() {
	LOG(LTRACE) << "Segmentation::step\n";
	m_clusters = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
	m_closed = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

	std::queue<cv::Point> open;
	open.push(cv::Point(320, 240));

	cv::Point right(1, 0);
	cv::Point left(-1, 0);
	cv::Point up(0, -1);
	cv::Point down(0, 1);

	while (!open.empty()) {
		cv::Point curpoint = open.front();
		open.pop();
		m_clusters.at<uchar>(curpoint) = 255;
		if (check(curpoint, right))	open.push(curpoint+right);
		if (check(curpoint, left))	open.push(curpoint+left);
		if (check(curpoint, up))	open.push(curpoint+up);
		if (check(curpoint, down))	open.push(curpoint+down);
	}

/*	for (int r = 0; r < 480; ++r) {
		cv::Point3f * nptr = m_normals.ptr<cv::Point3f>(r);
		cv::Point3f * dptr = m_depth.ptr<cv::Point3f>(r);
		uchar * optr = m_clusters.ptr<uchar>(r);
		for (int c = 0; c < 640-1; ++c) {
			float val = 180. / 3.14 * acos( nptr[c].dot(nptr[c+1]) );
			if ( (val < 3) && (norm(dptr[c]-dptr[c+1]) < 0.02) )
				optr[c] = 255;
			else
				optr[c] = 0;
		}
	}*/

	out_img.write(m_clusters);
	newImage->raise();

	return true;
}

bool Segmentation::onStop() {
	return true;
}

bool Segmentation::onStart() {
	return true;
}

void Segmentation::onNewDepth() {
	LOG(LTRACE) << "New depth";
	m_depth = in_depth.read().clone();
	m_depth_ready = true;

	if (m_depth_ready && m_normals_ready)
		onStep();
}

void Segmentation::onNewNormals() {
	LOG(LTRACE) << "New normals";
	m_normals = in_normals.read().clone();
	m_normals_ready = true;

	if (m_depth_ready && m_normals_ready)
		onStep();

}

} //: namespace Segmentation
} //: namespace Processors
