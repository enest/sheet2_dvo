// This source code is intended for use in the teaching course "Vision-Based Navigation" in summer term 2015 at TU Munich only. 
// Copyright 2015 Robert Maier, Joerg Stueckler, TUM

#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "cv_bridge/cv_bridge.h"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "opencv2/opencv.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/transforms.h>

#include <tf/transform_listener.h>


#include <sheet3_dvo/dvo.h>

#include <fstream>

#include <Eigen/Geometry>
#include <ctime>


cv::Mat grayRef, depthRef;
ros::Publisher pub_pointcloud;
tf::TransformListener *tfListener;
Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
void imagesToPointCloud( const cv::Mat& img_rgb, const cv::Mat& img_depth, pcl::PointCloud< pcl::PointXYZRGB >::Ptr& cloud, unsigned int downsampling = 1 ) {

  cloud->is_dense = true;
  cloud->height = img_depth.rows / downsampling;
  cloud->width = img_depth.cols / downsampling;
  cloud->sensor_origin_ = Eigen::Vector4f( 0.f, 0.f, 0.f, 1.f );
  cloud->sensor_orientation_ = Eigen::Quaternionf::Identity();
  cloud->points.resize( cloud->height*cloud->width );

  const float invfocalLength = 1.f / 525.f;
  const float centerX = 319.5f;
  const float centerY = 239.5f;
  const float depthscale = 1.f;

  const float* depthdata = reinterpret_cast<const float*>( &img_depth.data[0] );
  const unsigned char* colordata = &img_rgb.data[0];
  int idx = 0;
  for( unsigned int y = 0; y < img_depth.rows; y++ ) {
    for( unsigned int x = 0; x < img_depth.cols; x++ ) {

      if( x % downsampling != 0 || y % downsampling != 0 ) {
        colordata += 3;
        depthdata++;
        continue;
      }

      pcl::PointXYZRGB& p = cloud->points[idx];

      if( *depthdata == 0.f || std::isnan(*depthdata) ) { //|| factor * (float)(*depthdata) > 10.f ) {
        p.x = std::numeric_limits<float>::quiet_NaN();
        p.y = std::numeric_limits<float>::quiet_NaN();
        p.z = std::numeric_limits<float>::quiet_NaN();
      }
      else {
        float xf = x;
        float yf = y;
        float dist = depthscale * (float)(*depthdata);
        p.x = (xf-centerX) * dist * invfocalLength;
        p.y = (yf-centerY) * dist * invfocalLength;
        p.z = dist;
      }

      depthdata++;

      int b = (*colordata++);

      int g = (*colordata++);
      int r = (*colordata++);

      int rgb = ( r << 16 ) + ( g << 8 ) + b;
      p.rgb = * ( reinterpret_cast< float* > ( &rgb ) );

      idx++;


    }
  }

}

bool dumpTraj(const std::string &filename, const Eigen::Quaternionf q, const Eigen::Vector3f t)
{
    std::ofstream trajFile;
    trajFile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
    if (!trajFile.is_open())
        return false;

    //'timestamp tx ty tz qx qy qz qw'

    std::time_t timestamp = std::time(nullptr);

    trajFile << timestamp  << " "
             << t[0] << " "
             << t[1] << " "
             << t[2] << " "
             << q.x() << " "
             << q.y() << " "
             << q.z() << " "
             << q.w() << " "
             << std::endl;


    trajFile.close();

    return true;
}

bool dumpTraj(const std::string &filename, const Eigen::Matrix4f &transform)
{
    std::ofstream trajFile;
    trajFile.open(filename.c_str(), std::ofstream::out | std::ofstream::app);
    if (!trajFile.is_open())
        return false;

    //'timestamp tx ty tz qx qy qz qw'

    Eigen::Matrix3f rot;
    Eigen::Vector3f t;

    rot = transform.block<3,3>(0,0);
    t = transform.block<3,1>(0,3);
    Eigen::Quaternionf temp(rot);
    std::time_t timestamp = std::time(nullptr);

    trajFile << timestamp  << " "
             << t[0] << " "
             << t[1] << " "
             << t[2] << " "
             << temp.x() << " "
             << temp.y() << " "
             << temp.z() << " "
             << temp.w() << " "
             << std::endl;


    trajFile.close();

    /*ROS_ERROR_STREAM(  timestamp  << " "
                      << t[0] << " "
                      << t[1] << " "
                      << t[2] << " "
                      << temp.x() << " "
                      << temp.y() << " "
                      << temp.z() << " "
                      << temp.w() << " "
                      << std::endl );*/

    return true;
}



void callback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr& image_depth)
{
  
    Eigen::Matrix3f cameraMatrix;
    cameraMatrix <<    525.0, 0.0, 319.5,
                         0.0, 525.0, 239.5,
                         0.0, 0.0, 1.0;
    
    cv_bridge::CvImageConstPtr img_rgb_cv_ptr = cv_bridge::toCvShare( image_rgb, "bgr8" );
    cv_bridge::CvImageConstPtr img_depth_cv_ptr = cv_bridge::toCvShare( image_depth, "32FC1" );
    
    //cv::imshow("img_rgb", img_rgb_cv_ptr->image );
    //cv::imshow("img_depth", 0.2*img_depth_cv_ptr->image );
    //cv::waitKey(10);
    
    //Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    cv::Mat grayCurInt;
    cv::cvtColor( img_rgb_cv_ptr->image.clone(), grayCurInt, CV_BGR2GRAY);
    cv::Mat grayCur;
    grayCurInt.convertTo(grayCur, CV_32FC1, 1.f/255.f);
    
    cv::Mat depthCur = img_depth_cv_ptr->image.clone();
    
    
    if( !grayRef.empty() )
      alignImages( transform, grayRef, depthRef, grayCur, depthCur, cameraMatrix );
    
    grayRef = grayCur.clone();
    depthRef = depthCur.clone();
    
    ROS_ERROR_STREAM( "transform: " << transform << std::endl );
    

    // TODO: dump trajectory for evaluation
    if(!dumpTraj(std::string("/home/matiasvc/traj.txt"), transform))
        ROS_ERROR_STREAM( "Couldn't open the text file for dumping trajectories!" << std::endl );


    pcl::PointCloud< pcl::PointXYZRGB >::Ptr cloud = pcl::PointCloud< pcl::PointXYZRGB >::Ptr( new pcl::PointCloud< pcl::PointXYZRGB > );
    imagesToPointCloud( img_rgb_cv_ptr->image, img_depth_cv_ptr->image, cloud );
    
    cloud->header = pcl_conversions::toPCL( image_rgb->header );
    Eigen::Matrix4f integratedTransform;
    cloud->header.frame_id = "/world";
    pcl::transformPointCloud( *cloud, *cloud, integratedTransform );
        
    pub_pointcloud.publish( *cloud );

    tf::StampedTransform t;
    try
    {
        tfListener->waitForTransform("/world", "/openni_depth_optical_frame", image_rgb->header.stamp, ros::Duration(1.0f));
        tfListener->lookupTransform("/world", "/openni_depth_optical_frame", image_rgb->header.stamp, t);
    }
    catch (tf::TransformException ex)
    {
        ROS_ERROR("%s", ex.what());
        return;
    }

    const tf::Vector3 groundT = t.getOrigin();
    const tf::Quaternion groundQ = t.getRotation();

    const Eigen::Quaternionf eigQ(groundQ.x(), groundQ.y(), groundQ.z(), groundQ.w());
    const Eigen::Vector3f eigT(groundT.x(), groundT.y(), groundT.z());

    if(!dumpTraj(std::string("/home/matiasvc/traj-ground.txt"), eigQ, eigT))
        ROS_ERROR_STREAM( "Couldn't open the text file for dumping ground trajectories!" << std::endl );
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sheet2_dvo_node");

  ros::NodeHandle nh("~");
  message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, "image_rgb", 1);
  message_filters::Subscriber<sensor_msgs::Image> image_depth_sub(nh, "image_depth", 1);

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_rgb_sub, image_depth_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2));
  
  pub_pointcloud = nh.advertise< pcl::PointCloud< pcl::PointXYZRGB > >( "pointcloud", 1 );
  tfListener = new tf::TransformListener();


    ros::Rate loop_rate(100);

  while (ros::ok())
  {
    ros::spinOnce();

    loop_rate.sleep();
  }

  return 0;
}




