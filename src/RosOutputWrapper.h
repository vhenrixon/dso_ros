/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/* ======================= Some typical usecases: ===============
 *
 * (1) always get the pose of the most recent frame:
 *     -> Implement [publishCamPose].
 *
 * (2) always get the depthmap of the most recent keyframe
 *     -> Implement [pushDepthImageFloat] (use inverse depth in [image], and pose / frame information from [KF]).
 *
 * (3) accumulate final model
 *     -> Implement [publishKeyframes] (skip for final!=false), and accumulate frames.
 *
 * (4) get evolving model in real-time
 *     -> Implement [publishKeyframes] (update all frames for final==false).
 *
 *
 *
 *
 * ==================== How to use the structs: ===================
 * [FrameShell]: minimal struct kept for each frame ever tracked.
 *      ->camToWorld = camera to world transformation
 *      ->poseValid = false if [camToWorld] is invalid (only happens for frames during initialization).
 *      ->trackingRef = Shell of the frame this frame was tracked on.
 *      ->id = ID of that frame, starting with 0 for the very first frame.
 *
 *      ->incoming_id = ID passed into [addActiveFrame( ImageAndExposure* image, int id )].
 *	->timestamp = timestamp passed into [addActiveFrame( ImageAndExposure* image, int id )] as image.timestamp.
 *
 * [FrameHessian]
 *      ->immaturePoints: contains points that have not been "activated" (they do however have a depth initialization).
 *      ->pointHessians: contains active points.
 *      ->pointHessiansMarginalized: contains marginalized points.
 *      ->pointHessiansOut: contains outlier points.
 *
 *      ->frameID: incremental ID for keyframes only.
 *      ->shell: corresponding [FrameShell] struct.
 *
 *
 * [CalibHessian]
 *      ->fxl(), fyl(), cxl(), cyl(): get optimized, most recent (pinhole) camera intrinsics.
 *
 *
 * [PointHessian]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_scaled: inverse depth of point.
 *                       DO NOT USE [idepth], since it may be scaled with [SCALE_IDEPTH] ... however that is currently set to 1 so never mind.
 *      ->host: pointer to host-frame of point.
 *      ->status: current status of point (ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED)
 *      ->numGoodResiduals: number of non-outlier residuals supporting this point (approximate).
 *      ->maxRelBaseline: value roughly proportional to the relative baseline this point was observed with (0 = no baseline).
 *                        points for which this value is low are badly contrained.
 *      ->idepth_hessian: hessian value (inverse variance) of inverse depth.
 *
 * [ImmaturePoint]
 * 	->u,v: pixel-coordinates of point.
 *      ->idepth_min, idepth_max: the initialization sais that the inverse depth of this point is very likely
 *        between these two thresholds (their mean being the best guess)
 *      ->host: pointer to host-frame of point.
 */


#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "angles/angles.h"

#include <Eigen/Geometry>

#include "cv_bridge/cv_bridge.h"

#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

namespace dso
{
class FrameHessian;
class CalibHessian;
class FrameShell;

namespace IOWrap
{

  // some arbitrary values (0.1m^2 linear cov. 10deg^2. angular cov.)
  static const boost::array<double, 36> STANDARD_POSE_COVARIANCE =
  { { 0.1, 0, 0, 0, 0, 0,
      0, 0.1, 0, 0, 0, 0,
      0, 0, 0.1, 0, 0, 0,
      0, 0, 0, 0.17, 0, 0,
      0, 0, 0, 0, 0.17, 0,
      0, 0, 0, 0, 0, 0.17 } };
  static const boost::array<double, 36> STANDARD_TWIST_COVARIANCE =
  { { 0.002, 0, 0, 0, 0, 0,
      0, 0.002, 0, 0, 0, 0,
      0, 0, 0.05, 0, 0, 0,
      0, 0, 0, 0.09, 0, 0,
      0, 0, 0, 0, 0.09, 0,
      0, 0, 0, 0, 0, 0.09 } };
  static const boost::array<double, 36> BAD_COVARIANCE =
  { { 9999, 0, 0, 0, 0, 0,
      0, 9999, 0, 0, 0, 0,
      0, 0, 9999, 0, 0, 0,
      0, 0, 0, 9999, 0, 0,
      0, 0, 0, 0, 9999, 0,
  0, 0, 0, 0, 0, 9999 } };


class RosOutputWrapper : public Output3DWrapper
{
private:
//  ros::NodeHandle nh;
  ros::Publisher pub_pose;
  ros::Publisher pub_image;
  ros::Publisher pub_pointcloud;

  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener tfListener;
  geometry_msgs::TransformStamped transformStamped;

  std::string base_link_id;
  std::string frame_id;

  uint32_t sequence = 0;
  bool show_nonfinal_kf = false;    // create pointcloud data for frames which are not keyframes

public:
        inline RosOutputWrapper() : tfListener(tfBuffer)
        {
            ROS_INFO("Created RosOutputWrapper");
            int argc = 0;
            ros::init(argc, NULL, "dso_ros");
            ros::NodeHandle nh;
            ros::param::get("show_nonfinal_kf", show_nonfinal_kf);
            ROS_INFO("Showing nonfinal keyframes?  %s", show_nonfinal_kf ? "TRUE" : "FALSE");
            pub_pose = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>("/dso_ros/pose", 1000);
            pub_image = nh.advertise<sensor_msgs::Image>("/dso_ros/image", 100);
            pub_pointcloud = nh.advertise<PointCloud>("/dso_ros/pointcloud", 100);

            nh.param<std::string>("base_link_id", base_link_id, "base_link");
            nh.param<std::string>("frame_id", frame_id, "odomcamera_dso");
        }

        virtual ~RosOutputWrapper()
        {
            ROS_INFO("Destroyed RosOutputWrapper");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            ROS_INFO("Got graph with %d edges", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
                printf("Example Edge %d -> %d has %d active and %d marg residuals", idHost, idTarget, p.second[0], p.second[1]);
                maxWrite--;
                if(maxWrite==0) break;
            }
        }

        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {
            float fx = HCalib->fxl();
            float fy = HCalib->fyl();
            float cx = HCalib->cxl();
            float cy = HCalib->cyl();
            float fxi = 1/fx;
            float fyi = 1/fy;
            float cxi = -cx / fx;
            float cyi = -cy / fy;

            // output points to text file
//            std::ofstream output_points;
//            output_points.open("points.ply", std::ios_base::app);

            PointCloud::Ptr msg (new PointCloud);
            msg->height = msg->width = 1;    // width will get overwritten later
            msg->is_dense = false;        // for some reason it was defaulting true, which is not correct

            int counter = 0;
            for(FrameHessian* f : frames)
            {
                if (final || show_nonfinal_kf)
                {
                  auto const & m =  f->shell->camToWorld.matrix3x4();
                  auto const & points = f->pointHessiansMarginalized;
                  for (auto const * p : points) {
                      float depth = 1.0f / p->idepth;
                      auto const x = (p->u * fxi + cxi) * depth;
                      auto const y = (p->v * fyi + cyi) * depth;
                      auto const z = depth * (1 + 2*fxi);
                      Eigen::Vector4d camPoint(x, y, z, 1.f);
                      Eigen::Vector3d worldPoint = m * camPoint;
                      pcl::PointXYZ temp;
                      temp.x = worldPoint[0];
                      temp.y = worldPoint[1];
                      temp.z = worldPoint[2];
                      msg->points.push_back(temp);
//                      output_points << worldPoint.transpose() << std::endl;
                      counter++;
                  }
                  msg->width = counter;
                  std_msgs::Header header;
                  header.seq = sequence++ - 1;
                  header.stamp = ros::Time::now();
                  header.frame_id = frame_id;
                  pcl_conversions::toPCL(header, msg->header);
                  pub_pointcloud.publish(msg);
                  ROS_INFO("Published point cloud.");
  //                output_points.close();
                }
            }
        }

        // Convert matrix3x4 to ROS Pose
        /*
        geometry_msgs::Pose eigenMatrixToPoseMsg(const Eigen::MatrixXd& s) {
          geometry_msgs::Pose pose;
          Eigen::Vector3d translation = s.col(3);
          Eigen::Vector3f translationf = translation.cast<float>();
          pose.position = sophus_ros_conversions::eigenToPointMsg(translationf);
          Eigen::Matrix3d mat3d = s.block<3,3>(0,0);
          Eigen::Matrix3f mat3f = mat3d.cast<float>();
          Eigen::Quaternionf quaternionf(mat3f) ;
          pose.orientation = sophus_ros_conversions::eigenToQuaternionMsg(quaternionf);
          return pose;eigenMatrixToPoseMsg/DuowenQian/dso_ros/blob/master/src/ROSOutputPublisher.cpp
        */
        geometry_msgs::Pose matrixToPose(const Eigen::Matrix<Sophus::SE3Group<double>::Scalar,3,4> m ) {
            geometry_msgs::Pose pose;
            pose.position.x = m(0,3); 
            pose.position.y = m(1,3);
            pose.position.z = m(2,3);

            // camera orientation:  http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/ 
            double numX = 1 + m(0,0) - m(1,1) - m(2,2);
            double numY = 1 - m(0,0) + m(1,1) - m(2,2);
            double numZ = 1 - m(0,0) - m(1,1) + m(2,2);
            double numW = 1 + m(0,0) + m(1,1) + m(2,2);
            double camSX = sqrt( std::max( 0.0, numX ) ) / 2; 
            double camSY = sqrt( std::max( 0.0, numY ) ) / 2;
            double camSZ = sqrt( std::max( 0.0, numZ ) ) / 2;
            double camSW = sqrt( std::max( 0.0, numW ) ) / 2;
            geometry_msgs::Quaternion q;
            q.x = camSX;
            q.y = camSY;
            q.z = camSZ;
            q.w = camSW; // = geometry_msgs::Quaternion(camSX, camSY, camSZ, camSW);
            pose.orientation = q;
            pose.orientation = q;
            return pose;
        }

        

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            geometry_msgs::Pose pose = matrixToPose(frame->camToWorld.matrix3x4());
            //geometry_msgs::Pose pose = eigenMatrixToPoseMsg(frame->camToWorld.matrix3x4());
            //  Coordinates are in image frame: x-axis is on image columns, y-axis is on image lines, z axis is "forward" (depth)
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = frame_id; //     0 = no frame, 1 = global frame

            geometry_msgs::PoseWithCovariance pwc;
            pwc.pose = pose;
            pwc.covariance = STANDARD_POSE_COVARIANCE; // * .001; //
            geometry_msgs::PoseWithCovarianceStamped pwcs;
            header.stamp = ros::Time::now();
            pwcs.header = header;
            pwcs.pose = pwc;

            // try{
            //     transformStamped = tfBuffer.lookupTransform(base_link_id, frame_id, ros::Time(0));
            //     ROS_INFO("Got the transform.");
            // }
            // catch (tf2::TransformException &ex) {
            //     ROS_WARN("%s",ex.what());
            //     ros::Duration(1.0).sleep();
            // }
            // tf2::doTransform(pwcs, pwcs, transformStamped);

            pub_pose.publish(pwcs);
            ROS_INFO("Published PoseWithCovarianceStamped.");
        }


        virtual void pushLiveFrame(FrameHessian* image) override
        {
//            ROS_INFO("pushLiveFrame\n");
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
          // publishes input image overlayed with key points
          sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv::Mat(image->h, image->w, CV_8UC3, image->data)).toImageMsg();
          pub_image.publish(msg);
          ROS_INFO("Published Depth Image.");
        }

        virtual bool needPushDepthImage() override
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
            ROS_INFO("Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:",
                   KF->frameID,
                   KF->shell->incoming_id,
                   KF->shell->timestamp,
                   KF->shell->id);
            std::cout << KF->shell->camToWorld.matrix3x4() << "\n";

            int maxWrite = 5;
            for(int y=0;y<image->h;y++)
            {
                for(int x=0;x<image->w;x++)
                {
                    if(image->at(x,y) <= 0) continue;

                    ROS_INFO("Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
        }


};



}



}