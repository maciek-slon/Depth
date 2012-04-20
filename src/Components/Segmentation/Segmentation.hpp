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

#include <cv.h>

#include <pcl/point_types.h>

namespace Processors {
namespace Segmentation {

/*!
 * \class Segmentation
 * \brief Segmentation processor class.
 */
class Segmentation: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	Segmentation(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~Segmentation();


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
	Base::EventHandler <Segmentation> h_onNewDepth;

	/// Event handler.
	Base::EventHandler <Segmentation> h_onNewNormals;

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_depth;

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_normals;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_img;

	Base::Property<float> prop_ang_diff;
	Base::Property<float> prop_dist_diff;
	Base::Property<float> prop_std_diff;

private:

	void onNewDepth();
	void onNewNormals();

	bool check(cv::Point point, cv::Point dir);
	bool newSeed(cv::Point point, cv::Point dir);

	cv::Mat m_normals;
	cv::Mat m_depth;

	cv::Mat m_clusters;
	cv::Mat m_closed;

	bool m_normals_ready;
	bool m_depth_ready;
};

}//: namespace Segmentation
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("Segmentation", Processors::Segmentation::Segmentation, Common::Panel_Empty)

#endif /* SEGMENTATION_HPP_ */
