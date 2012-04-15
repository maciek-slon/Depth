/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>

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

bool Segmentation::onStep() {
	LOG(LTRACE) << "Segmentation::step\n";
	m_clusters.create(cv::Size(640, 480), CV_8UC1);

	for (int r = 0; r < 480; ++r) {
		cv::Point3f * nptr = m_normals.ptr<cv::Point3f>(r);
		uchar * optr = m_clusters.ptr<uchar>(r);
		for (int c = 0; c < 640-1; ++c) {
			float val = 180. / 3.14 * acos( nptr[c].dot(nptr[c+1]) );
			if (val < 1)
				optr[c] = 255;
			else if (val < 3)
				optr[c] = 128;
			else
				optr[c] = 0;
		}
	}

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
