/*!
 * \file
 * \brief 
 * \author Łukasz Żmuda
 */

#ifndef DEPTHTRANSFORM_HPP_
#define DEPTHTRANSFORM_HPP_

#include "Component_Aux.hpp"
#include "Component.hpp"
#include "DataStream.hpp"
#include "Property.hpp"
#include "EventHandler2.hpp"

#include <Types/CameraInfo.hpp>
#include <opencv2/opencv.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Types/HomogMatrix.hpp"
/*#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
*/

namespace Processors {
namespace DepthTransform {


/*!
 * \class DepthTransform
 * \brief DepthTransform processor class.
 *
 * DepthTransform processor.
 */
class DepthTransform: public Base::Component {
public:
	/*!
	 * Constructor.
	 */
	DepthTransform(const std::string & name = "DepthTransform");

	/*!
	 * Destructor
	 */
	virtual ~DepthTransform();

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
	Base::DataStreamIn <cv::Mat, Base::DataStreamBuffer::Newest> in_image_xyz;
	Base::DataStreamIn <Types::HomogMatrix, Base::DataStreamBuffer::Newest> in_homogMatrix;


	// Output data streams
	Base::DataStreamOut<cv::Mat> out_image_xyz;	
/*        Base::DataStreamOut<pcl::PointCloud<pcl::PointXYZ>::Ptr > out_cloud_xyz_start;
	Base::DataStreamOut<pcl::PointCloud<pcl::PointXYZ>::Ptr > out_cloud_xyz_stop;*/


	/// Property - performs inverse transformation.
	Base::Property<bool> prop_inverse;

	/// Property - if set, no transformation is applied.
        Base::Property<bool> pass_through;

	// Handlers
	void DepthTransformation();

private:
	/// Maximal range <-300,300> (in meters).
	static const double MAX_RANGE = 300;

	/// Invalid coordinate.
	static const float INVALID_COORDINATE = 100000;

};

} //: namespace DepthTransform
} //: namespace Processors

/*
 * Register processor component.
 */
REGISTER_COMPONENT("DepthTransform", Processors::DepthTransform::DepthTransform)

#endif /* DEPTHTRANSFORM_HPP_ */
