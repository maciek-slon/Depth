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
#include "Property.hpp"

#include <string>

#include <opencv2/core/core.hpp>

namespace Processors {
namespace NormalEstimator {

enum Algorithm {

};

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

	void onNewImage();

	/// Event handler.
	Base::EventHandler <NormalEstimator> h_onNewImage;

	/// Input data stream
	Base::DataStreamIn <cv::Mat, Base::DataStreamBuffer::Newest> in_img;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_img;

	/// Output data stream - processed image
	Base::DataStreamOut <cv::Mat> out_normals;

	Base::Property<float> prop_radius;

private:
	cv::Mat img;
	cv::Mat out;
	cv::Mat normals;

	Algorithm m_algorithm;
};

}//: namespace NormalEstimator
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_PROCESSOR_COMPONENT("NormalEstimator", Processors::NormalEstimator::NormalEstimator, Common::Panel_Empty)

#endif /* NORMALESTIMATOR_HPP_ */
