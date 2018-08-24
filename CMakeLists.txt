cmake_minimum_required(VERSION 2.4.6)
project(dso_ros)
add_compile_options(-std=c++11)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Release)

rosbuild_init()

#rosbuild_add_boost_directories()

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)


set(DSO_PATH $ENV{DSO_PATH})
message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")

find_package(sophus_ros_conversions REQUIRED)
message("DENTAL PLAN... Lisa needs braces...")

find_package(catkin REQUIRED COMPONENTS roscpp rospy geometry_msgs)

catkin_package(
  CATKIN_DEPENDS geometry_msgs roscpp
)

include_directories(include ${catkin_INCLUDE_DIRS})

find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)

# SSE flags
rosbuild_check_for_sse()
set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer -lboost_system "
)

set(SOURCE_FILES
  src/main.cpp
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  /opt/ros/kinetic/share/sophus_ros_conversions/
)

rosbuild_add_executable(dso_live src/main.cpp ${SOURCE_FILES})
target_link_libraries(dso_live ${DSO_LIBRARY} ${Pangolin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(dso_live boost_system)
target_link_libraries(dso_live ${catkin_LIBRARIES})
target_link_libraries(dso_live sophus_ros_conversions)
rosbuild_link_boost(dso_live thread)
