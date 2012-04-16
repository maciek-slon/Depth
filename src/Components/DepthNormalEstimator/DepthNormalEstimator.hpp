/*!
 * \file
 * \brief
 */

#ifndef DEPTHNORMALESTIMATOR_HPP_
#define DEPTHNORMALESTIMATOR_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"

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
	Base::EventHandler <NormalEstimator> h_onNewImage;

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_img;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_img;

	/// Event raised, when image is processed
	Base::Event * newNormals;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_normals;

private:
	cv::Mat img;
	cv::Mat out;
	cv::Mat normals;
};

}//: namespace DepthNormalEstimator
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("DepthNormalEstimator", Processors::DepthNormalEstimator::DepthNormalEstimator, Common::Panel_Empty)

#endif /* DEPTHNORMALESTIMATOR_HPP_ */
