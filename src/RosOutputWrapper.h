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


#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"

#include "sophus_ros_conversions/geometry.hpp"
#include "sophus_ros_conversions/eigen.hpp"
#include <Eigen/Geometry>

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

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

class RosOutputWrapper : public Output3DWrapper
{
private:
//  ros::NodeHandle nh;
  ros::Publisher pub_pose;
  ros::Publisher pub_image;
  ros::Publisher pub_pointcloud;
  uint32_t sequence = 0;
  bool show_nonfinal_kf = false;    // create pointcloud data for frames which are not keyframes

public:
        inline RosOutputWrapper()    //const ros::Publisher& publisher
        {
            printf("OUT: Created RosOutputWrapper\n");
            int argc = 0;
            ros::init(argc, NULL, "dso_ros");
            ros::NodeHandle nh;
            pub_pose = nh.advertise<geometry_msgs::PoseStamped>("/dso_ros/pose", 1000);
            pub_image = nh.advertise<sensor_msgs::Image>("/dso_ros/image", 100);
            pub_pointcloud = nh.advertise<PointCloud>("/dso_ros/pointcloud", 100);
        }

        virtual ~RosOutputWrapper()
        {
            printf("OUT: Destroyed RosOutputWrapper\n");
        }

        virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i, std::less<uint64_t>, Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>> &connectivity) override
        {
            printf("OUT: got graph with %d edges\n", (int)connectivity.size());

            int maxWrite = 5;

            for(const std::pair<uint64_t,Eigen::Vector2i> &p : connectivity)
            {
                int idHost = p.first>>32;
                int idTarget = p.first & ((uint64_t)0xFFFFFFFF);
                printf("OUT: Example Edge %d -> %d has %d active and %d marg residuals\n", idHost, idTarget, p.second[0], p.second[1]);
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
                  header.frame_id = "pointcloud_frame";
                  pcl_conversions::toPCL(header, msg->header);
                  pub_pointcloud.publish(msg);
  //                output_points.close();
                }
            }
        }

        // Convert matrix3x4 to ROS Pose
        geometry_msgs::Pose eigenMatrixToPoseMsg(const Eigen::MatrixXd& s) {
          geometry_msgs::Pose pose;
          Eigen::Vector3d translation = s.col(3);
          Eigen::Vector3f translationf = translation.cast<float>();
          pose.position = sophus_ros_conversions::eigenToPointMsg(translationf);
          Eigen::Matrix3d mat3d = s.block<3,3>(0,0);
          Eigen::Matrix3f mat3f = mat3d.cast<float>();
          Eigen::Quaternionf quaternionf(mat3f) ;
          pose.orientation = sophus_ros_conversions::eigenToQuaternionMsg(quaternionf);
          return pose;
        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
            geometry_msgs::Pose pose = eigenMatrixToPoseMsg(frame->camToWorld.matrix3x4());
            //  Coordinates are in image frame: x-axis is on image columns, y-axis is on image lines, z axis is "forward" (depth)
            std_msgs::Header header;
            header.stamp = ros::Time::now();
            header.frame_id = "1"; //     0 = no frame, 1 = global frame
            geometry_msgs::PoseStamped ps;
            ps.header = header;
            ps.pose = pose;
            pub_pose.publish(ps);
            ROS_INFO("Published pose.");
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
        }

        virtual bool needPushDepthImage() override
        {
            return false;
        }

        virtual void pushDepthImageFloat(MinimalImageF* image, FrameHessian* KF ) override
        {
            printf("OUT: Predicted depth for KF %d (id %d, time %f, internal frame-ID %d). CameraToWorld:\n",
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

                    printf("OUT: Example Idepth at pixel (%d,%d): %f.\n", x,y,image->at(x,y));
                    maxWrite--;
                    if(maxWrite==0) break;
                }
                if(maxWrite==0) break;
            }
        }


};



}



}
