/*!
 * \file
 * \brief
 */

#ifndef DEPTHMAPGENERATOR_HPP_
#define DEPTHMAPGENERATOR_HPP_

#include "Base/Component_Aux.hpp"
#include "Base/Component.hpp"
#include "Base/DataStream.hpp"

#include <opencv2/core/core.hpp>

namespace Processors {
namespace DepthMapGenerator {

/*!
 * \class DepthMapGenerator
 * \brief DepthMapGenerator processor class.
 */
class DepthMapGenerator: public Base::Component
{
public:
	/*!
	 * Constructor.
	 */
	DepthMapGenerator(const std::string & name = "");

	/*!
	 * Destructor
	 */
	virtual ~DepthMapGenerator();

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

	/// Output data stream - generated image
	Base::DataStreamOut <cv::Mat> out_img;


	cv::Mat m_image;
	int m_width;
	int m_height;
};

}//: namespace DepthMapGenerator
}//: namespace Processors


/*
 * Register processor component.
 */
REGISTER_COMPONENT("DepthMapGenerator", Processors::DepthMapGenerator::DepthMapGenerator)

#endif /* DEPTHMAPGENERATOR_HPP_ */
