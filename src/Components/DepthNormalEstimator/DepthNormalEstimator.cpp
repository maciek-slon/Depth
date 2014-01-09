/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>

#include "DepthNormalEstimator.hpp"
#include "Common/Logger.hpp"

#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>

namespace Processors {
namespace DepthNormalEstimator {

DepthNormalEstimator::DepthNormalEstimator(const std::string & name) :
		Base::Component(name),
		prop_difference_threshold("difference_threshold", 20) {
	LOG(LTRACE)<< "Hello DepthNormalEstimator\n";

	registerProperty(prop_difference_threshold);
}

DepthNormalEstimator::~DepthNormalEstimator() {
	LOG(LTRACE)<< "Good bye DepthNormalEstimator\n";
}

void DepthNormalEstimator::prepareInterface() {
	h_onNewImage.setup(this, &DepthNormalEstimator::onNewImage);

	registerStream("in_depth", &in_img);
	registerStream("out_img", &out_img);
	registerStream("out_normals", &out_normals);

	registerHandler("onNewImage", &h_onNewImage);
	addDependency("onNewImage", &in_img);

}

bool DepthNormalEstimator::onInit() {
	LOG(LTRACE)<< "DepthNormalEstimator::initialize\n";

	return true;
}

bool DepthNormalEstimator::onFinish() {
	LOG(LTRACE)<< "DepthNormalEstimator::finish\n";

	return true;
}

bool DepthNormalEstimator::onStep() {
	LOG(LTRACE)<< "DepthNormalEstimator::step\n";
	return true;
}

bool DepthNormalEstimator::onStop() {
	return true;
}

bool DepthNormalEstimator::onStart() {
	return true;
}

static void accumBilateral(long delta, long i, long j, long * A, long * b,
		int threshold) {
	long f = std::abs(delta) < threshold ? 1 : 0;

	const long fi = f * i;
	const long fj = f * j;

	A[0] += fi * i;
	A[1] += fi * j;
	A[3] += fj * j;
	b[0] += fi * delta;
	b[1] += fj * delta;
}

void DepthNormalEstimator::onNewImage() {
	img = in_img.read();
	cv::Mat tmp = cv::Mat::zeros(img.size(), CV_8U);
	out = cv::Mat::zeros(img.size(), CV_8UC3);
	normals = cv::Mat::zeros(img.size(), CV_32FC3);

	long distance_threshold = 2000;
	int difference_threshold = prop_difference_threshold;

	IplImage src_ipl = img;
	IplImage* ap_depth_data = &src_ipl;
	IplImage dst_ipl = tmp;
	IplImage* dst_ipl_ptr = &dst_ipl;

	unsigned short * lp_depth = (unsigned short *) ap_depth_data->imageData;

	const int l_W = ap_depth_data->width;
	const int l_H = ap_depth_data->height;

	const int l_r = 5; // used to be 7
	const int l_offset0 = -l_r - l_r * l_W;
	const int l_offset1 = 0 - l_r * l_W;
	const int l_offset2 = +l_r - l_r * l_W;
	const int l_offset3 = -l_r;
	const int l_offset4 = +l_r;
	const int l_offset5 = -l_r + l_r * l_W;
	const int l_offset6 = 0 + l_r * l_W;
	const int l_offset7 = +l_r + l_r * l_W;

	for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y) {
		unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);

		for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x) {
			long l_d = lp_line[0];

			if (l_d < distance_threshold) {
				// accum
				long l_A[4];
				l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
				long l_b[2];
				l_b[0] = l_b[1] = 0;
				accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset1] - l_d, 0, -l_r, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset3] - l_d, -l_r, 0, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset4] - l_d, +l_r, 0, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset6] - l_d, 0, +l_r, l_A, l_b,
						difference_threshold);
				accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b,
						difference_threshold);

				// solve
				long l_det = l_A[0] * l_A[3] - l_A[1] * l_A[1];
				long l_ddx = l_A[3] * l_b[0] - l_A[1] * l_b[1];
				long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

				/// @todo Magic number 1150 is focal length? This is something like
				/// f in SXGA mode, but in VGA is more like 530.
				float l_nx = static_cast<float>(530 * l_ddx);
				float l_ny = static_cast<float>(530 * l_ddy);
				float l_nz = static_cast<float>(-l_det * l_d);

				float l_sqrt = sqrt(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

				if (l_sqrt > 0) {
					float l_norminv = 1.0f / (l_sqrt);

					l_nx *= l_norminv;
					l_ny *= l_norminv;
					l_nz *= l_norminv;

					normals.at<cv::Point3f>(l_y, l_x) = cv::Point3f(-l_nx,
							-l_ny, -l_nz);

				} else {
					normals.at<cv::Point3f>(l_y, l_x) = cv::Point3f(-1, -1, -1);
				}
			} else {
				normals.at<cv::Point3f>(l_y, l_x) = cv::Point3f(-1, -1, -1);
			}
			++lp_line;
		}
	}
	//cvSmooth(m_dep[0], m_dep[0], CV_MEDIAN, 5, 5);
	cv::convertScaleAbs(normals, out, 128, 128);
	cv::cvtColor(out, out, CV_RGB2BGR);

	out_img.write(out.clone());

	out_normals.write(normals.clone());
}

} //: namespace DepthNormalEstimator
} //: namespace Processors
