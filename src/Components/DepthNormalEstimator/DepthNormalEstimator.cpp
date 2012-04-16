/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>

#include "DepthNormalEstimator.hpp"
#include "Common/Logger.hpp"

namespace Processors {
namespace DepthNormalEstimator {

DepthNormalEstimator::DepthNormalEstimator(const std::string & name) : Base::Component(name)
{
	LOG(LTRACE) << "Hello DepthNormalEstimator\n";
}

DepthNormalEstimator::~DepthNormalEstimator()
{
	LOG(LTRACE) << "Good bye DepthNormalEstimator\n";
}

bool DepthNormalEstimator::onInit()
{
	LOG(LTRACE) << "DepthNormalEstimator::initialize\n";

	// Register data streams, events and event handlers HERE!

	return true;
}

bool DepthNormalEstimator::onFinish()
{
	LOG(LTRACE) << "DepthNormalEstimator::finish\n";

	return true;
}

bool DepthNormalEstimator::onStep()
{
	LOG(LTRACE) << "DepthNormalEstimator::step\n";
	return true;
}

bool DepthNormalEstimator::onStop()
{
	return true;
}

bool DepthNormalEstimator::onStart()
{
	return true;
}

}//: namespace DepthNormalEstimator
}//: namespace Processors
