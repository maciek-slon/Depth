/*!
 * \file
 * \brief
 */

#ifndef NORMALESTIMATOR_HPP_
#define NORMALESTIMATOR_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "Panel_Empty.hpp"
#include "DataStream.hpp"

#include <cv.h>

namespace Processors {
namespace NormalEstimator {

/*!
 * \class NormalEstimator
 * \brief NormalEstimator processor class.
 */
class NormalEstimator: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	NormalEstimator(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~NormalEstimator();


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

	void onNewImage();

	/// Event handler.
	Base::EventHandler <NormalEstimator> h_onNewImage;

	/// Input data stream
	Base::DataStreamIn <cv::Mat> in_img;

	/// Event raised, when image is processed
	Base::Event * newImage;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_img;


private:
	cv::Mat img;
	cv::Mat out;

	std::vector<uint16_t> m_gamma;
};

}//: namespace NormalEstimator
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("NormalEstimator", Processors::NormalEstimator::NormalEstimator, Common::Panel_Empty)

#endif /* NORMALESTIMATOR_HPP_ */
