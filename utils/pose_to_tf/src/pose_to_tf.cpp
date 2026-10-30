#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

class PoseToTF : public rclcpp::Node
{
public:
  PoseToTF()
  : Node("pose_to_tf")
  {
    this->declare_parameter<std::string>("parent_frame", "odom");
    this->declare_parameter<std::string>("child_frame", "base_link");
    this->declare_parameter<std::string>("pose_topic", "pose");
    // empty disables the odometry republisher
    this->declare_parameter<std::string>("odom_in_topic", "");
    this->declare_parameter<std::string>("odom_topic", "odom");

    parent_frame_ = this->get_parameter("parent_frame").as_string();
    child_frame_ = this->get_parameter("child_frame").as_string();
    std::string pose_topic = this->get_parameter("pose_topic").as_string();
    std::string odom_in_topic = this->get_parameter("odom_in_topic").as_string();

    subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      pose_topic, 10, std::bind(&PoseToTF::topic_callback, this, std::placeholders::_1));
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    if (!odom_in_topic.empty()) {
      odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        this->get_parameter("odom_topic").as_string(), 10);
      odom_subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
        odom_in_topic, 10, std::bind(&PoseToTF::odom_callback, this, std::placeholders::_1));
    }
  }

private:
  void topic_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    geometry_msgs::msg::TransformStamped transform_stamped;

    transform_stamped.header.stamp = msg->header.stamp;
    transform_stamped.header.frame_id = parent_frame_;
    transform_stamped.child_frame_id = child_frame_;

    transform_stamped.transform.translation.x = msg->pose.position.x;
    transform_stamped.transform.translation.y = msg->pose.position.y;
    transform_stamped.transform.translation.z = msg->pose.position.z;
    transform_stamped.transform.rotation = msg->pose.orientation;

    tf_broadcaster_->sendTransform(transform_stamped);
    pose_ = msg;
  }

  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!pose_) {
      return;
    }
    msg->header.frame_id = parent_frame_;
    msg->child_frame_id = child_frame_;
    msg->pose.pose = pose_->pose;
    odom_publisher_->publish(*msg);
  }

  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr subscription_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  geometry_msgs::msg::PoseStamped::SharedPtr pose_;
  std::string parent_frame_;
  std::string child_frame_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PoseToTF>());
  rclcpp::shutdown();
  return 0;
}
