cmake_minimum_required(VERSION 3.8)
project(gnss_denied_vio_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(vio_ekf_wrapper_node src/vio_ekf_wrapper_node.cpp)
target_compile_features(vio_ekf_wrapper_node PUBLIC cxx_std_17)
target_link_libraries(vio_ekf_wrapper_node Eigen3::Eigen)
ament_target_dependencies(
  vio_ekf_wrapper_node
  geometry_msgs
  nav_msgs
  rclcpp
  sensor_msgs
)

install(TARGETS vio_ekf_wrapper_node DESTINATION lib/${PROJECT_NAME})

ament_package()

