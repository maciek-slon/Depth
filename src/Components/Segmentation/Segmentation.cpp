/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>
#include <queue>
#include <cstdlib>
#include <functional>
#include <numeric>

#include "Segmentation.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#include <opencv2/imgproc/imgproc.hpp>

namespace Processors {
namespace Segmentation {


double compareNormals(unsigned char* v1, unsigned char* v2, double n) {
	cv::Point3f * curn = (cv::Point3f*)v1;
	cv::Point3f * desn = (cv::Point3f*)v2;
	double dn = 180. / 3.14 * acos(curn->dot(*desn));
	dn = (dn < 180 ? dn : 0);
	return dn / n;
}

double compareColors(unsigned char* v1, unsigned char* v2, double n) {
	typedef cv::Point3_<uchar> Point3u;
	Point3u *curc = (Point3u*)v1;
	Point3u *desc = (Point3u*)v2;

	cv::Point3f distc = *desc;
	cv::Point3f distc2 = *curc;
	distc -= distc2;
	distc *= 1. / 255;

	return norm(distc) / n;
}

double comparePositions(unsigned char* v1, unsigned char* v2, double n) {
	cv::Point3f *curp = (cv::Point3f*)v1;
	cv::Point3f *desp = (cv::Point3f*)v2;
	double dp = norm(*desp - *curp);
	dp = (dp < 10 ? dp : 0);
	return dp / n;
}

double accumulateSum(std::vector<double> values) {
	return std::accumulate(values.begin(), values.end(), 0);
}

double accumulateMax(std::vector<double> values) {
	return *std::max_element(values.begin(), values.end());
}

Segmentation::Segmentation(const std::string & name) :
		Base::Component(name),
		prop_ang_diff("ang_diff", 2.0f),
		prop_dist_diff("dist_diff", 0.02f),
		prop_color_diff("color_diff", 2.0f),
		prop_std_diff("std_diff", 2.0f),
		prop_threshold("threshold", 3.0f) {
	LOG(LTRACE)<< "Hello Segmentation\n";
	m_normals_ready = m_depth_ready = m_color_ready = false;

	registerProperty(prop_ang_diff);
	registerProperty(prop_dist_diff);
	registerProperty(prop_color_diff);
	registerProperty(prop_std_diff);
	registerProperty(prop_threshold);
}

Segmentation::~Segmentation() {
	LOG(LTRACE)<< "Good bye Segmentation\n";
}

void Segmentation::prepareInterface() {
	// Register data streams, events and event handlers HERE!

	h_onNewDepth.setup(this, &Segmentation::onNewDepth);
	registerHandler("onNewDepth", &h_onNewDepth);

	h_onNewColor.setup(this, &Segmentation::onNewColor);
	registerHandler("onNewColor", &h_onNewColor);

	h_onNewNormals.setup(this, &Segmentation::onNewNormals);
	registerHandler("onNewNormals", &h_onNewNormals);

	registerStream("in_or depth", &in_depth);
	registerStream("in_color", &in_color);
	registerStream("in_normals", &in_normals);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	h_onColor.setup(boost::bind(&Segmentation::onNewData, this, true, false, false));
	registerHandler("onColor", &h_onColor);
	addDependency("onColor", &in_color);

	h_onColorDepth.setup(boost::bind(&Segmentation::onNewData, this, true, true, false));
	registerHandler("onColorDepth", &h_onColorDepth);
	addDependency("onColorDepth", &in_color);
	addDependency("onColorDepth", &in_depth);

	h_onColorDepthNormals.setup(boost::bind(&Segmentation::onNewData, this, true, true, true));
	registerHandler("onColorDepethNormals", &h_onColorDepthNormals);
	addDependency("onColorDepthNormalas", &in_color);
	addDependency("onColorDepthNormals", &in_depth);
	addDependency("onColorDepthNormals", &in_normals);

	h_onDepth.setup(boost::bind(&Segmentation::onNewData, this, false, true, false));
	registerHandler("onDepth", &h_onDepth);
	addDependency("onDepth", &in_depth);

	h_onDepthNormals.setup(boost::bind(&Segmentation::onNewData, this, false, true, true));
	registerHandler("onDepthNormals", &h_onDepthNormals);
	addDependency("onDepthNormals", &in_depth);
	addDependency("onDepthNormals", &in_normals);
}

bool Segmentation::onInit() {
	LOG(LTRACE)<< "Segmentation::initialize\n";

	return true;
}

bool Segmentation::onFinish() {
	LOG(LTRACE)<< "Segmentation::finish\n";

	return true;
}

bool Segmentation::check(cv::Point point, cv::Point dir) {
	typedef cv::Point3_<uchar> Point3u;

	float tn = prop_ang_diff;
	float tp = prop_dist_diff;
	float tc = prop_color_diff;
	float ts = prop_threshold;

	cv::Point dest = point + dir;

	if (!dest.inside(cv::Rect(0, 0, 639, 479)))
		return false;

	if (m_closed.at<uchar>(dest) == 255)
		return false;

	m_closed.at<uchar>(dest) = 255;

	cv::Point3f curn = m_normals.at<cv::Point3f>(point);
	cv::Point3f curd = m_depth.at<cv::Point3f>(point);
	Point3u curc = m_color.at<Point3u>(point);
	cv::Point3f desn = m_normals.at<cv::Point3f>(point + dir);
	cv::Point3f desd = m_depth.at<cv::Point3f>(point + dir);
	Point3u desc = m_color.at<Point3u>(point + dir);

	float dn = 180. / 3.14 * acos(curn.dot(desn));
	dn = dn < 180 ? dn : 0;
	float dp = norm(desd - curd);
	dp = dp < 10 ? dp : 0;

	cv::Point3f distc = desc;
	cv::Point3f distc2 = curc;
	distc -= distc2;
	distc *= 1. / 255;

	float dc = norm(distc);

	float difference = dc / tc + dp / tp + dn / tn;

	return difference < ts;
}

bool Segmentation::check(cv::Point point, cv::Point dir,
		std::vector<cv::Mat> inputs, std::vector<Comparator> comparators,
		std::vector<double> thresholds, Accumulator accumulator, double threshold) {

	cv::Point dest = point + dir;

	// check, if given direction lays inside image
	// TODO: check real size of image
	if (!dest.inside(cv::Rect(0, 0, 639, 479)))
		return false;

	// ignore already segmented points
	if (m_closed.at<uchar>(dest) == 255)
		return false;

	// mark point as segmented
	m_closed.at<uchar>(dest) = 255;

	// intermediate results
	std:cumulate:vector<double> results;

	// iterate over all available inputs
	for (int i = 0; i < inputs.size(); ++i) {
		cv::Mat img = inputs[i];
		unsigned char * v1 = img.data + point.y * img.step + point.x * img.elemSize();
		unsigned char * v2 = img.data + dest.y * img.step + dest.x * img.elemSize();
		results.push_back(comparators[i](v1, v2, thresholds[i]));
	}

	// accumulate intermediate results
	double result = accumulator(results);

	return result < threshold;
}

cv::Mat Segmentation::multimodalSegmentation(std::vector<cv::Mat> inputs,
		std::vector<Comparator> comparators, std::vector<double> thresholds,
		Accumulator accumulator, double threshold) {


	typedef cv::Point3_<uchar> CvColor;

	m_clusters = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
	m_closed = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

	CvColor empty(0, 0, 0);

	std::queue<cv::Point> open;
	std::queue<cv::Point> seed;

	// initialize seeds in regular 10x10 grid
	for (int x = 0; x < 640; x += 10)
		for (int y = 0; y < 480; y += 10)
			seed.push(cv::Point(x, y));

	// definition of all possible directions
	cv::Point right(1, 0);
	cv::Point left(-1, 0);
	cv::Point up(0, -1);
	cv::Point down(0, 1);

	// repeat until we still have some seed points
	while (!seed.empty()) {
		// create new, empty list of open points
		open = std::queue<cv::Point>();

		// get first seed
		cv::Point pt = seed.front();
		seed.pop();

		// ignore already segmented seeds
		if (m_clusters.at<CvColor>(pt) != empty) {
			continue;
		}

		// generate random color for new segment
		cv::Point3i id(0, rand() % 128, rand() % 128);

		cv::Point3f point_mean(0, 0, 0);
		open.push(pt);
		float acc = 0;
		float acc2 = 0;
		float angle;

		LOG(LDEBUG)<< "Growing";
		// growing segment
		while (!open.empty()) {

			cv::Point curpoint = open.front();
			open.pop();
			if (m_clusters.at<CvColor>(curpoint) != empty)
				continue;

			m_clusters.at<CvColor>(curpoint) = id;

			if (check(curpoint, right, inputs, comparators, thresholds, accumulator, threshold))
				open.push(curpoint + right);
			if (check(curpoint, left, inputs, comparators, thresholds, accumulator, threshold))
				open.push(curpoint + left);
			if (check(curpoint, up, inputs, comparators, thresholds, accumulator, threshold))
				open.push(curpoint + up);
			if (check(curpoint, down, inputs, comparators, thresholds, accumulator, threshold))
				open.push(curpoint + down);
		}
	}

	return m_clusters;
}

void Segmentation::onNewData(bool color, bool depth, bool normals) {
	std::vector<cv::Mat> inputs;
	std::vector<Comparator> comparators;
	std::vector<double> thresholds;

	if (color) {
		inputs.push_back(in_color.read().clone());
		comparators.push_back(compareColors);
		thresholds.push_back(prop_color_diff);
	}
	if (depth) {
		inputs.push_back(in_depth.read().clone());
		comparators.push_back(comparePositions);
		thresholds.push_back(prop_dist_diff);
	}
	if (normals) {
		inputs.push_back(in_normals.read().clone());
		comparators.push_back(compareNormals);
		thresholds.push_back(prop_ang_diff);
	}

	cv::Mat ret = multimodalSegmentation(inputs, comparators, thresholds, accumulateSum, prop_threshold);

	out_img.write(ret.clone());
}

bool Segmentation::newSeed(cv::Point point, cv::Point dir) {
	return true;
}

bool Segmentation::onStep() {
	try {
		m_depth_ready = m_normals_ready = m_color_ready = false;
		typedef cv::Point3_<uchar> CvColor;
		LOG(LDEBUG)<< "Segmentation::step\n";
		m_clusters = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
		m_closed = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

		CvColor empty(0, 0, 0);

		std::queue<cv::Point> open;
		std::queue<cv::Point> seed;

		for (int x = 0; x < 640; x += 10)
			for (int y = 0; y < 480; y += 10)
				seed.push(cv::Point(x, y));

		cv::Point right(1, 0);
		cv::Point left(-1, 0);
		cv::Point up(0, -1);
		cv::Point down(0, 1);

		srand(0);

		std::queue<cv::Point3f> blob_normals;
		std::queue<cv::Point3f> blob_points;
		while (!seed.empty()) {

			open = std::queue<cv::Point>();

			cv::Point pt = seed.front();
			seed.pop();
			// ignore already segmented seeds
			if (m_clusters.at<CvColor>(pt) != empty) {
				continue;
			}

			cv::Point3i id(0, rand() % 128, rand() % 128);
			int size = 0;
			cv::Point3f point_mean(0, 0, 0);
			open.push(pt);
			float acc = 0;
			float acc2 = 0;
			float angle;

			LOG(LDEBUG)<< "Growing";
			// growing segment
			while (!open.empty()) {

				cv::Point curpoint = open.front();
				open.pop();
				if (m_clusters.at<CvColor>(curpoint) != empty)
					continue;

				cv::Point3f p = m_depth.at<cv::Point3f>(curpoint);
				point_mean += p;

				blob_normals.push(m_normals.at<cv::Point3f>(curpoint));
				blob_points.push(p);

				m_clusters.at<CvColor>(curpoint) = id;
				size++;
				if (check(curpoint, right))
					open.push(curpoint + right);
				if (check(curpoint, left))
					open.push(curpoint + left);
				if (check(curpoint, up))
					open.push(curpoint + up);
				if (check(curpoint, down))
					open.push(curpoint + down);
			}

			//if (blob_normals.size() > 1)
			LOG(LDEBUG)<< "Normals: " << blob_normals.size();

			// calculating features for segment
			point_mean *= 1.0f / size;

			acc = acc2 = 0;
			while (!blob_normals.empty()) {
				// normalized vector from point to center of mass
				cv::Point3f ntc = point_mean - blob_points.front();
				ntc *= 1.0f / norm(ntc);
				// normal in current point
				cv::Point3f nor = blob_normals.front();
				// angle between both vectors
				angle = 180. / 3.14 * acos(nor.dot(ntc));
				acc += angle;
				acc2 += angle * angle;
				blob_normals.pop();
				blob_points.pop();
			}

			LOG(LDEBUG)<< "Calculated";

			// calculate mean angle and it's deviation
			float mean = acc / size;
			float std_dev = sqrt(
					((size * acc2) - (acc * acc)) / (size * (size - 1)));

			/*if (std_dev > 20)
			 cv::floodFill(m_clusters, pt, cv::Scalar(mean, mean, mean));
			 else if (mean < 70)
			 cv::floodFill(m_clusters, pt, cv::Scalar(mean * 1.5, 0, 0));
			 else if (mean < 100)
			 cv::floodFill(m_clusters, pt, cv::Scalar(0, mean * 1.5, 0));
			 else
			 cv::floodFill(m_clusters, pt, cv::Scalar(0, 0, mean * 1.5));*/

			LOG(LDEBUG)<< "Flooded";
		}

		LOG(LDEBUG)<< "Finishing";

		cv::medianBlur(m_clusters, m_clusters, 5);

		out_img.write(m_clusters.clone());
		newImage->raise();
	} catch (...) {
		LOG(LERROR)<< "Segmentation::onStep failed\n";
	}
	return true;
}

bool Segmentation::onStop() {
	return true;
}

bool Segmentation::onStart() {
	return true;
}

void Segmentation::onNewDepth() {
	LOG(LTRACE)<< "New depth";
	m_depth = in_depth.read().clone();
	m_depth_ready = true;

	if (m_depth_ready && m_normals_ready && m_color_ready) onStep();
}

void Segmentation::onNewColor() {
	LOG(LTRACE)<< "New color";
	m_color = in_color.read().clone();
	m_color_ready = true;

	if (m_depth_ready && m_normals_ready && m_color_ready) onStep();
}

void Segmentation::onNewNormals() {
	LOG(LTRACE)<< "New normals";
	m_normals = in_normals.read().clone();
	m_normals_ready = true;

	if (m_depth_ready && m_normals_ready && m_color_ready) onStep();
}

} //: namespace Segmentation
} //: namespace Processors
