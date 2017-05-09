/*!
 * \file
 * \brief 
 * \author Maciej Stefanczyk
 */

#ifndef PASSTHROUGH_HPP_
#define PASSTHROUGH_HPP_

#include "Base/Component_Aux.hpp"
#include "Base/Component.hpp"
#include "Base/DataStream.hpp"
#include "Base/Property.hpp"
#include "Base/EventHandler2.hpp"

#include <opencv2/opencv.hpp>


namespace Processors {
namespace PassThrough {

/*!
 * \class PassThrough
 * \brief PassThrough processor class.
 *
 * 
 */
class PassThrough: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	PassThrough(const std::string & name = "PassThrough");

	/*!
	 * Destructor
	 */
	virtual ~PassThrough();

	/*!
	 * Prepare components interface (register streams and handlers).
	 * At this point, all properties are already initialized and loaded to 
	 * values set in config file.
	 */
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
	 * Start component
	 */
	bool onStart();

	/*!
	 * Stop component
	 */
	bool onStop();


	// Input data streams
	Base::DataStreamIn<cv::Mat> in_xyz;

	// Output data streams
	Base::DataStreamOut<cv::Mat> out_xyz;
	Base::DataStreamOut<cv::Mat> out_mask;

	// Handlers

	// Properties
	Base::Property<float> z_min;
	Base::Property<float> z_max;

	
	// Handlers
	void onNewImage();

};

} //: namespace PassThrough
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("PassThrough", Processors::PassThrough::PassThrough)

#endif /* PASSTHROUGH_HPP_ */
