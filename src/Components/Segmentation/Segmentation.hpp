/*!
 * \file
 * \brief
 */

#ifndef SEGMENTATION_HPP_
#define SEGMENTATION_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include <opencv2/core/core.hpp>

namespace Processors {
namespace Segmentation {

typedef double (*Comparator)(unsigned char *, unsigned char *, double);

double compareNormals(unsigned char * v1, unsigned char * v2, double n);
double compareColors(unsigned char * v1, unsigned char * v2, double n);
double comparePositions(unsigned char * v1, unsigned char * v2, double n);

typedef double (*Accumulator)(std::vector<double>);

double accumulateSum(std::vector<double> values);
double accumulateMax(std::vector<double> values);

/*!
 * \class Segmentation
 * \brief Segmentation processor class.
 */
class Segmentation: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	Segmentation(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~Segmentation();

	void prepareInterface();

protected:

	/*!
	 * Connects source to given device.
	 */
	bool onInit();

	/*!
	 * Disconnect source from device, closes streams, etc.
	 */
	bool onFinish();

	/*!
	 * Retrieves data from device.
	 */
	bool onStep();

	/*!
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();

	/// Event handler.
	Base::EventHandler<Segmentation> h_onNewDepth;

	/// Event handler.
	Base::EventHandler<Segmentation> h_onNewColor;

	/// Event handler.
	Base::EventHandler<Segmentation> h_onNewNormals;

	/// Event handlers
	Base::EventHandler2 h_onColor;

	Base::EventHandler2 h_onColorDepth;

	Base::EventHandler2 h_onColorDepthNormals;

	Base::EventHandler2 h_onDepth;

	Base::EventHandler2 h_onDepthNormals;

	/// Input data stream
	Base::DataStreamIn<cv::Mat> in_depth;

	/// Input data stream
	Base::DataStreamIn<cv::Mat> in_color;

	/// Input data stream
	Base::DataStreamIn<cv::Mat> in_normals;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - processed image
	Base::DataStreamOut<cv::Mat> out_img;

	// Tc
	Base::Property<float> prop_color_diff;

	// Tp
	Base::Property<float> prop_dist_diff;

	// Tn
	Base::Property<float> prop_ang_diff;

	// Ts
	Base::Property<float> prop_threshold;

	Base::Property<float> prop_std_diff;

private:

	void onNewDepth();
	void onNewColor();
	void onNewNormals();

	void onNewData(bool color, bool depth, bool normals);

	cv::Mat multimodalSegmentation(std::vector<cv::Mat> inputs, std::vector<Comparator> comparators, std::vector<double> thresholds, Accumulator accumulator, double threshold);
	bool check(cv::Point point, cv::Point dir, std::vector<cv::Mat> inputs, std::vector<Comparator> comparators, std::vector<double> thresholds, Accumulator accumulator, double threshold);

	bool check(cv::Point point, cv::Point dir);
	bool newSeed(cv::Point point, cv::Point dir);

	cv::Mat m_normals;
	cv::Mat m_depth;
	cv::Mat m_color;

	cv::Mat m_clusters;
	cv::Mat m_closed;

	bool m_normals_ready;
	bool m_depth_ready;
	bool m_color_ready;
};

} //: namespace Segmentation
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("Segmentation",
		Processors::Segmentation::Segmentation, Common::Panel_Empty)

#endif /* SEGMENTATION_HPP_ */
