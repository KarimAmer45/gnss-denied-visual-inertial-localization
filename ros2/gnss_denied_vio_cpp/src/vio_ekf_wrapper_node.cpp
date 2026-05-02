#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>

namespace {

constexpr int kStateSize = 5;
constexpr int kX = 0;
constexpr int kY = 1;
constexpr int kVx = 2;
constexpr int kVy = 3;
constexpr int kYaw = 4;

double wrap_angle(double angle) {
  while (angle >= M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

double yaw_from_quaternion(const geometry_msgs::msg::Quaternion & q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

geometry_msgs::msg::Quaternion quaternion_from_yaw(double yaw) {
  geometry_msgs::msg::Quaternion q;
  q.w = std::cos(yaw * 0.5);
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  return q;
}

}  // namespace

class VioEkfWrapperNode final : public rclcpp::Node {
 public:
  VioEkfWrapperNode() : Node("vio_ekf_wrapper_node") {
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
      "/imu/data", rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) { handle_imu(*msg); });

    wheel_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/wheel/odom", 20,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) { handle_wheel(*msg); });

    visual_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/visual_odometry", 20,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) { handle_visual(*msg); });

    gnss_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/gnss/pose", 10,
      [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { handle_gnss(*msg); });

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/localization/ekf_odom", 20);
    covariance_ = Eigen::Matrix<double, kStateSize, kStateSize>::Identity() * 4.0;
  }

 private:
  void handle_imu(const sensor_msgs::msg::Imu & msg) {
    if (!last_imu_stamp_.nanoseconds()) {
      last_imu_stamp_ = rclcpp::Time(msg.header.stamp);
      return;
    }

    const rclcpp::Time stamp(msg.header.stamp);
    const double dt = std::max(1e-3, (stamp - last_imu_stamp_).seconds());
    last_imu_stamp_ = stamp;

    const double yaw = state_(kYaw);
    const double c = std::cos(yaw);
    const double s = std::sin(yaw);
    const double ax = c * msg.linear_acceleration.x - s * msg.linear_acceleration.y;
    const double ay = s * msg.linear_acceleration.x + c * msg.linear_acceleration.y;

    state_(kX) += state_(kVx) * dt + 0.5 * ax * dt * dt;
    state_(kY) += state_(kVy) * dt + 0.5 * ay * dt * dt;
    state_(kVx) += ax * dt;
    state_(kVy) += ay * dt;
    state_(kYaw) = wrap_angle(state_(kYaw) + msg.angular_velocity.z * dt);

    Eigen::Matrix<double, kStateSize, kStateSize> f =
      Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
    f(kX, kVx) = dt;
    f(kY, kVy) = dt;
    Eigen::Matrix<double, kStateSize, kStateSize> q =
      Eigen::Matrix<double, kStateSize, kStateSize>::Zero();
    q(kX, kX) = 0.02 * dt * dt;
    q(kY, kY) = 0.02 * dt * dt;
    q(kVx, kVx) = 0.12 * dt;
    q(kVy, kVy) = 0.12 * dt;
    q(kYaw, kYaw) = 0.01 * dt;
    covariance_ = f * covariance_ * f.transpose() + q;
    publish(msg.header.stamp);
  }

  void handle_wheel(const nav_msgs::msg::Odometry & msg) {
    const double speed = msg.twist.twist.linear.x;
    const double yaw = state_(kYaw);
    Eigen::Matrix<double, 1, kStateSize> h;
    h.setZero();
    h(0, kVx) = std::cos(yaw);
    h(0, kVy) = std::sin(yaw);
    h(0, kYaw) = -std::sin(yaw) * state_(kVx) + std::cos(yaw) * state_(kVy);
    const double predicted = std::cos(yaw) * state_(kVx) + std::sin(yaw) * state_(kVy);
    Eigen::Matrix<double, 1, 1> r;
    r << 0.08 * 0.08;
    update<1>(Eigen::Matrix<double, 1, 1>::Constant(speed - predicted), h, r);
    publish(msg.header.stamp);
  }

  void handle_visual(const nav_msgs::msg::Odometry & msg) {
    const double yaw = yaw_from_quaternion(msg.pose.pose.orientation);
    Eigen::Matrix<double, 3, 1> residual;
    residual << msg.pose.pose.position.x - state_(kX), msg.pose.pose.position.y - state_(kY),
      wrap_angle(yaw - state_(kYaw));

    Eigen::Matrix<double, 3, kStateSize> h;
    h.setZero();
    h(0, kX) = 1.0;
    h(1, kY) = 1.0;
    h(2, kYaw) = 1.0;
    Eigen::Matrix3d r = Eigen::Vector3d(0.65 * 0.65, 0.65 * 0.65, 0.05 * 0.05).asDiagonal();
    update<3>(residual, h, r);
    publish(msg.header.stamp);
  }

  void handle_gnss(const geometry_msgs::msg::PoseWithCovarianceStamped & msg) {
    Eigen::Matrix<double, 2, 1> residual;
    residual << msg.pose.pose.position.x - state_(kX), msg.pose.pose.position.y - state_(kY);

    Eigen::Matrix<double, 2, kStateSize> h;
    h.setZero();
    h(0, kX) = 1.0;
    h(1, kY) = 1.0;
    Eigen::Matrix2d r = Eigen::Vector2d(1.2 * 1.2, 1.2 * 1.2).asDiagonal();
    update<2>(residual, h, r);
    publish(msg.header.stamp);
  }

  template <int Rows>
  void update(
    const Eigen::Matrix<double, Rows, 1> & residual,
    const Eigen::Matrix<double, Rows, kStateSize> & h,
    const Eigen::Matrix<double, Rows, Rows> & r) {
    const Eigen::Matrix<double, Rows, Rows> s = h * covariance_ * h.transpose() + r;
    const Eigen::Matrix<double, kStateSize, Rows> k = covariance_ * h.transpose() * s.inverse();
    state_ += k * residual;
    state_(kYaw) = wrap_angle(state_(kYaw));
    const auto identity = Eigen::Matrix<double, kStateSize, kStateSize>::Identity();
    covariance_ = (identity - k * h) * covariance_ * (identity - k * h).transpose() + k * r * k.transpose();
  }

  void publish(const builtin_interfaces::msg::Time & stamp) {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";
    msg.child_frame_id = "base_link";
    msg.pose.pose.position.x = state_(kX);
    msg.pose.pose.position.y = state_(kY);
    msg.pose.pose.orientation = quaternion_from_yaw(state_(kYaw));
    msg.twist.twist.linear.x = std::cos(state_(kYaw)) * state_(kVx) + std::sin(state_(kYaw)) * state_(kVy);
    msg.twist.twist.angular.z = 0.0;
    odom_pub_->publish(msg);
  }

  Eigen::Matrix<double, kStateSize, 1> state_{Eigen::Matrix<double, kStateSize, 1>::Zero()};
  Eigen::Matrix<double, kStateSize, kStateSize> covariance_;
  rclcpp::Time last_imu_stamp_{0, 0, RCL_ROS_TIME};

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr wheel_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr visual_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr gnss_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VioEkfWrapperNode>());
  rclcpp::shutdown();
  return 0;
}
