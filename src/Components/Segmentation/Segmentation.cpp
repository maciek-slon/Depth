/*!
 * \file
 * \brief
 */

#include <memory>
#include <string>
#include <queue>
#include <cstdlib>

#include "Segmentation.hpp"
#include "Common/Logger.hpp"

namespace Processors {
namespace Segmentation {

Segmentation::Segmentation(const std::string & name) :
	Base::Component(name),
		prop_ang_diff("ang_diff", 2.0f),
		prop_dist_diff("dist_diff", 0.02f),
                prop_color_diff("color_diff", 2.0f),
		prop_std_diff("std_diff", 2.0f),
                prop_threshold("threshold", 3.0f) {
	LOG(LTRACE) << "Hello Segmentation\n";
	m_normals_ready = m_depth_ready = m_color_ready = false;

	registerProperty(prop_ang_diff);
	registerProperty(prop_dist_diff);
        registerProperty(prop_color_diff);
	registerProperty(prop_std_diff);
        registerProperty(prop_threshold);
}

Segmentation::~Segmentation() {
	LOG(LTRACE) << "Good bye Segmentation\n";
}

bool Segmentation::onInit() {
	LOG(LTRACE) << "Segmentation::initialize\n";

	// Register data streams, events and event handlers HERE!

	h_onNewDepth.setup(this, &Segmentation::onNewDepth);
	registerHandler("onNewDepth", &h_onNewDepth);

	h_onNewColor.setup(this, &Segmentation::onNewColor);
	registerHandler("onNewColor", &h_onNewColor);        

	h_onNewNormals.setup(this, &Segmentation::onNewNormals);
	registerHandler("onNewNormals", &h_onNewNormals);

	registerStream("in_depth", &in_depth);
        registerStream("in_color", &in_color);
	registerStream("in_normals", &in_normals);

	newImage = registerEvent("newImage");

	registerStream("out_img", &out_img);

	return true;
}

bool Segmentation::onFinish() {
	LOG(LTRACE) << "Segmentation::finish\n";

	return true;
}


bool Segmentation::check(cv::Point point, cv::Point dir) {
        float tn = prop_ang_diff;
        float tp = prop_dist_diff;
        float tc = prop_color_diff;
        float ts = prop_threshold;

        cv::Point dest = point+dir;
        
        if (!dest.inside(cv::Rect(0, 0, 639, 479)))
                return false;        
        
        if (m_closed.at<uchar>(dest) == 255)
                return false;

        m_closed.at<uchar>(dest) = 255;

        cv::Point3f curn = m_normals.at<cv::Point3f>(point);
        cv::Point3f curd = m_depth.at<cv::Point3f>(point);
        int curc = (int)(m_color.at<uchar>(point));
        cv::Point3f desn = m_normals.at<cv::Point3f>(point+dir);
        cv::Point3f desd = m_depth.at<cv::Point3f>(point+dir);
        int desc = (int)(m_color.at<uchar>(point+dir));
        
        float dn = 180. / 3.14 * acos( curn.dot(desn) );
        float dp = norm(desd-curd);
        int dc = abs(desc-curc);
        
        float difference = dc/tc + dp/tp + dn/tn;
        
        return difference < ts;
        // return ((angle < ang_diff && dist < dist_diff)); // || cdist < color_diff );
}

bool Segmentation::newSeed(cv::Point point, cv::Point dir) {
	return true;
}

bool Segmentation::onStep() {
    try {
            //pcl::PointCloud<pcl::PointXYZ> cloud;
            m_depth_ready = m_normals_ready = m_color_ready = false;
            typedef cv::Point3_<uchar> CvColor;
            LOG(LTRACE) << "Segmentation::step\n";
            m_clusters = cv::Mat::zeros(cv::Size(640, 480), CV_8UC3);
            m_closed = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);

            CvColor empty(0, 0, 0);

            std::queue<cv::Point> open;
            std::queue<cv::Point> seed;

            for (int x = 0; x < 640; x+=10)
                    for (int y = 0; y < 480; y+=10)
                            seed.push(cv::Point(x, y));

            cv::Point right(1, 0);
            cv::Point left(-1, 0);
            cv::Point up(0, -1);
            cv::Point down(0, 1);


            std::queue<cv::Point3f> blob_normals;
            std::queue<cv::Point3f> blob_points;
            while (!seed.empty()) {
                    open = std::queue<cv::Point>();

                    cv::Point pt = seed.front();
                    seed.pop();
                    // ignore already segmented seeds
                    if (m_clusters.at<CvColor>(pt) != empty) {
                            continue;
                    }

                    cv::Point3i id(0, rand()%128, rand() % 128);
                    int size = 0;
                    cv::Point3f point_mean(0, 0, 0);
                    open.push(pt);
                    float acc = 0;
                    float acc2 = 0;
                    float angle;
                    //std::vector<int> inliers;

                    // growing segment
                    while (!open.empty()) {
                            cv::Point curpoint = open.front();
                            open.pop();
                            if (m_clusters.at<CvColor>(curpoint) != empty)
                                    continue;

                            //angle = 180. / 3.14 * acos( m_normals.at<cv::Point3f>(curpoint).dot(cv::Point3f(0, 0, 1)) );
                            //acc += angle;
                            //acc2 += angle*angle;

                            cv::Point3f p = m_depth.at<cv::Point3f>(curpoint);
                            point_mean += p;

                            blob_normals.push(m_normals.at<cv::Point3f>(curpoint));
                            blob_points.push(p);

                            //cloud.push_back(pcl::PointXYZ(p.x, p.y, p.z));

                            m_clusters.at<CvColor>(curpoint) = id;
                            size++;
                            if (check(curpoint, right))	open.push(curpoint+right);
                            if (check(curpoint, left))	open.push(curpoint+left);
                            if (check(curpoint, up))	open.push(curpoint+up);
                            if (check(curpoint, down))	open.push(curpoint+down);
                    }

                    // calculating features for segment
                    point_mean *= 1.0f/size;

                    acc = acc2 = 0;
                    while (!blob_normals.empty()) {
                            // normalized vector from point to center of mass
                            cv::Point3f ntc = point_mean - blob_points.front();
                            ntc *= 1.0f / norm(ntc);
                            // normal in current point
                            cv::Point3f nor = blob_normals.front();
                            // angle between both vectors
                            angle = 180. / 3.14 * acos( nor.dot(ntc) );
                            acc += angle;
                            acc2 += angle*angle;
                            blob_normals.pop();
                            blob_points.pop();
                    }

                    // calculate mean angle and it's deviation
                    float mean = acc / size;
                    float std_dev = sqrt(((size * acc2) - (acc * acc)) / (size * (size - 1)));

                    if (std_dev > 20)
                            cv::floodFill(m_clusters, pt, cv::Scalar(mean, mean, mean));
                    else if (mean < 70)
                            cv::floodFill(m_clusters, pt, cv::Scalar(mean*1.5, 0, 0));
                    else if (mean < 100)
                            cv::floodFill(m_clusters, pt, cv::Scalar(0, mean*1.5, 0));
                    else
                            cv::floodFill(m_clusters, pt, cv::Scalar(0, 0, mean*1.5));

                    /*cv::Mat points(1, 1, CV_32FC3);
                    points.at<cv::Point3f>(0, 0) = point_mean;
                    std::vector<cv::Point2f> proj_points;
                    cv::Mat rot = cv::Mat::zeros(1, 3, CV_32FC1);
                    cv::Mat cam = cv::Mat::zeros(3, 3, CV_32FC1);
                    cam.at<float>(0, 0) = 1.0 / 5.9421434211923247e+02;
                    cam.at<float>(1, 1) = 1.0 / 5.9104053696870778e+02;
                    cam.at<float>(2, 2) = 1.0;
                    cam.at<float>(0, 2) = 320;
                    cam.at<float>(1, 2) = 240;

                    cv::projectPoints(points, rot, rot, cam, cv::Mat::zeros(1, 4, CV_32FC1), proj_points);
                    cv::Point target = proj_points[0];
                    cv::Point3f accum(0, 0, 0);

                    int cnt = 0;
                    for (int i=-1; i <= 1; ++i)
                            for (int j = -1; j <= 1; ++j) {
                                    cv::Point3f p = m_depth.at<cv::Point3f>(target + cv::Point(i, j));
                                    if (p.x < 10000) {
                                            accum += p;
                                            cnt ++;
                                    }
                            }

                    accum *= 1.0 / cnt;

                    float mean = acc / size;
                    float std_dev = sqrt(((size * acc2) - (acc * acc)) / (size * (size - 1)));
                    if (std_dev < prop_std_diff) {
                            cv::floodFill(m_clusters, pt, cv::Scalar(255, 0, 0));
                    } else if (norm(accum) < norm(point_mean)) {
                            cv::floodFill(m_clusters, pt, cv::Scalar(0, 255, 0));
                    } else if (norm(accum) > norm(point_mean)) {
                            cv::floodFill(m_clusters, pt, cv::Scalar(0, 0, 255));
                    }*/
            }

            cv::medianBlur(m_clusters, m_clusters, 5);

            out_img.write(m_clusters.clone());
            newImage->raise();
        } catch (...) {
            LOG(LERROR) << "Segmentation::onStep failed\n";
        }
	return true;
}

bool Segmentation::onStop() {
	return true;
}

bool Segmentation::onStart() {
	return true;
}

void Segmentation::onNewDepth() {
	LOG(LTRACE) << "New depth";
	m_depth = in_depth.read().clone();
	m_depth_ready = true;

	if (m_depth_ready && m_normals_ready && m_color_ready)
		onStep();
}

void Segmentation::onNewColor() {
	LOG(LTRACE) << "New color";
	m_color = in_color.read().clone();
	m_color_ready = true;
        
	if (m_depth_ready && m_normals_ready && m_color_ready)
		onStep();
}

void Segmentation::onNewNormals() {
	LOG(LTRACE) << "New normals";
	m_normals = in_normals.read().clone();
	m_normals_ready = true;

	if (m_depth_ready && m_normals_ready && m_color_ready)
		onStep();
}

} //: namespace Segmentation
} //: namespace Processors
