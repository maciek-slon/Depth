/*!
 * \file
 * \brief
 */

#ifndef DEPTHNORMALESTIMATOR_HPP_
#define DEPTHNORMALESTIMATOR_HPP_

#include "Base/Component_Aux.hpp"
#include "Base/Component.hpp"
#include "Base/DataStream.hpp"
#include "Base/Property.hpp"

#include <opencv2/core/core.hpp>

namespace Processors {
namespace DepthNormalEstimator {

/*!
 * \class DepthNormalEstimator
 * \brief DepthNormalEstimator processor class.
 */
class DepthNormalEstimator: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	DepthNormalEstimator(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~DepthNormalEstimator();

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
	Base::EventHandler <DepthNormalEstimator> h_onNewImage;

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_img;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_img;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_normals;

private:
	cv::Mat img;
	cv::Mat out;
	cv::Mat normals;

	Base::Property<int> prop_difference_threshold;

	void onNewImage();
};

}//: namespace DepthNormalEstimator
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_COMPONENT("DepthNormalEstimator", Processors::DepthNormalEstimator::DepthNormalEstimator)

#endif /* DEPTHNORMALESTIMATOR_HPP_ */
