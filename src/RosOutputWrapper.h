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
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include "cv_bridge/cv_bridge.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class RosOutputWrapper : public Output3DWrapper
{
public:
        inline RosOutputWrapper()
        {
            printf("OUT: Created RosOutputWrapper\n");
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


        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {
            for(FrameHessian* f : frames)
            {
                printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d immature points. CameraToWorld:\n",
                       f->frameID,
                       final ? "final" : "non-final",
                       f->shell->incoming_id,
                       f->shell->timestamp,
                       (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int)f->immaturePoints.size());
                std::cout << f->shell->camToWorld.matrix3x4() << "\n";


                int maxWrite = 5;
                for(PointHessian* p : f->pointHessians)
                {
                    printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
                           p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
                    maxWrite--;
                    if(maxWrite==0) break;
                }
            }
        }

        virtual void publishCamPose(FrameShell* frame, CalibHessian* HCalib) override
        {
//            printf("ROS OUT: Current Frame %d (time %f, internal ID %d). CameraToWorld:\n",
//                   frame->incoming_id,
//                   frame->timestamp,
//                   frame->id);
//            std::cout << frame->camToWorld.matrix3x4() << "\n";
//            geometry_msgs::Pose pose = sophus_ros_conversions::sophusToPoseMsg(frame->camToWorld.matrix3x4().cast<float>());
            geometry_msgs::Pose pose = eigenMatrixToPoseMsg(frame->camToWorld.matrix3x4());
//            geometry_msgs::Pose pose = sophusDoubleToPoseMsg(frame->camToWorld);
//              geometry_msgs::Pose pose = sophus_ros_conversions::sophusToPoseMsg(frame->camToWorld.cast<float>());
            printf("ROS POSE ::\n");
            std::cout << pose << "\n";
        }


        virtual void pushLiveFrame(FrameHessian* image) override
        {
            // can be used to get the raw image / intensity pyramid.
        }

        virtual void pushDepthImage(MinimalImageB3* image) override
        {
            // can be used to get the raw image with depth overlay.
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
