---
sidebar_position: 4
title: 'Chapter 3: Navigation & Path Planning (Nav2)'
---

Chapter 3: Navigation & Path Planning (Nav2)

## Learning Objectives

After completing this chapter, you will be able to:

- Install and configure Nav2 for humanoid robots with bipedal locomotion constraints
- Implement global and local planners adapted for bipedal navigation
- Configure Nav2 parameters specifically for humanoid robot dynamics
- Simulate bipedal-safe navigation scenarios using Nav2
- Integrate Nav2 with Isaac ROS for complete navigation solutions

## Key Topics

### 1. Nav2 Installation and Configuration for Humanoid Robots

- System requirements and prerequisites for Nav2
- Installing Nav2 packages with humanoid-specific dependencies
- Setting up ROS 2 environment for navigation
- Configuring Nav2 for bipedal locomotion constraints

### 2. Global and Local Planners for Bipedal Robots

- Understanding global planner algorithms for humanoid navigation
- Configuring local planners for bipedal-safe path execution
- Custom costmap layers for humanoid-specific navigation
- Humanoid-specific path optimization techniques

### 3. Nav2 Parameter Configuration for Bipedal Constraints

- Adjusting velocity and acceleration limits for bipedal locomotion
- Configuring balance-aware navigation parameters
- Setting appropriate safety margins for bipedal robots
- Tuning planners for humanoid kinematic constraints

### 4. Bipedal-Safe Navigation Scenarios

- Creating navigation scenarios that account for bipedal stability
- Implementing safe path planning around obstacles
- Handling dynamic environments with bipedal constraints
- Simulating navigation in various terrain conditions

### 5. Nav2 Integration with Isaac ROS

- Connecting Nav2 with Isaac ROS perception systems
- Implementing sensor fusion for navigation
- Creating complete perception-navigation pipeline
- Validating integrated systems for humanoid robots

## Practical Implementation

### Setting up Nav2 for Humanoid Robots

To configure Nav2 for humanoid robot navigation, you need to install the appropriate packages and configure the system for bipedal constraints:

1. **System Requirements**:
   - ROS 2 installation (Humble Hawksbill recommended)
   - Nav2 packages and dependencies
   - Humanoid robot model with appropriate URDF
   - Sensor configuration for navigation (lidar, cameras, IMU)

2. **Installation Process**:
   - Install ROS 2 Humble or newer
   - Install Nav2 packages: `sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup`
   - Install additional dependencies for humanoid navigation
   - Configure robot-specific parameters

3. **Initial Configuration**:
   - Create navigation configuration files specific to humanoid robot
   - Configure costmaps for bipedal navigation
   - Set up transforms and coordinate frames

### Nav2 Configuration for Bipedal Navigation

Here's a complete Nav2 configuration tailored for bipedal humanoid robots:

```yaml
# Navigation configuration for bipedal humanoid robot
bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml
    default_nav_to_pose_bt_xml: /opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_w_replanning_and_recovery.xml
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_are_error_positions_close_condition_bt_node
      - nav2_would_a_controller_recovery_help_condition_bt_node
      - nav2_am_i_oscillating_condition_bt_node
      - nav2_is_recovering_from_costmap_blockage_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_is_goal_reached_condition_bt_node
      - nav2_is_path_blocked_condition_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_recover_nav_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_single_trigger_bt_node
      - nav2_is_battery_charging_condition_bt_node

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 10.0  # Lower frequency for bipedal stability
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller with bipedal constraints
    FollowPath:
      plugin: "nav2_mppi_controller::MPPIController"
      time_horizon: 2.0  # Longer horizon for stability
      frequency: 10.0
      velocity_samples: 1
      model_dt: 0.1  # Slower updates for stability
      batch_size: 2000
      vx_std: 0.1  # Reduced for bipedal stability
      vy_std: 0.05
      wxy_std: 0.1  # Reduced for bipedal stability
      vx_max: 0.3  # Reduced max speed for bipedal safety
      vx_min: -0.1
      vy_max: 0.1
      wz_max: 0.3  # Reduced angular velocity for balance
      wz_min: -0.3
      model_noise: 0.05
      temperature: 0.3
      horizon_delay: 1
      control_horizon: 4
      xy_goal_tolerance: 0.3  # Larger tolerance for bipedal robots
      yaw_goal_tolerance: 0.3
      stateful: true
      motion_model: "DiffDrive"
      reference_tracker:
        k_phi: 1.5  # Reduced for stability
        k_delta: 1.0  # Reduced for stability
        k_vel: 0.8  # Reduced for stability
        k_omega: 0.3  # Reduced for stability
        track_error_scale: 1.0
        cmd_vel_scale: 1.0
        velocity_scaling_tolerance: 0.1
        velocity_scaling_min: 0.05
        max_velocity_scaling_factor: 1.0
        min_velocity_scaling_factor: 0.05

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      use_rollout_costs: True
      lethal_cost_threshold: 100
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05  # Higher resolution for detailed planning
      transform_tolerance: 0.5
      footprint: "[[-0.4, -0.3], [-0.4, 0.3], [0.4, 0.3], [0.4, -0.3]]"  # Larger for humanoid
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0  # Higher inflation for safety
        inflation_radius: 0.55  # Larger safety margin for bipedal robots
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 8
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
```

### Humanoid-Specific Path Planning Node

Here's an example of implementing a humanoid-specific path planning node:

```python
import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import math

class HumanoidPathPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_path_planner')

        # Create action client for navigation
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )

        # Bipedal-specific parameters
        self.max_step_height = 0.15  # Maximum step height for bipedal robot
        self.min_path_width = 0.6    # Minimum path width for safe bipedal navigation
        self.balance_margin = 0.2    # Safety margin for balance

        # Navigation goal publisher
        self.goal_publisher = self.create_publisher(
            PoseStamped, '/goal_pose', 10
        )

    def create_bipedal_safe_path(self, start_pose, goal_pose):
        """
        Create a path that considers bipedal locomotion constraints
        """
        # Calculate direct path
        dx = goal_pose.pose.position.x - start_pose.pose.position.x
        dy = goal_pose.pose.position.y - start_pose.pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate intermediate waypoints considering bipedal constraints
        waypoints = []

        # Add intermediate waypoints for smoother navigation
        num_waypoints = max(5, int(distance / 0.5))  # At least 5 waypoints, 0.5m spacing

        for i in range(1, num_waypoints):
            t = i / num_waypoints
            waypoint = PoseStamped()
            waypoint.header = Header()
            waypoint.header.stamp = self.get_clock().now().to_msg()
            waypoint.header.frame_id = "map"

            waypoint.pose.position.x = start_pose.pose.position.x + t * dx
            waypoint.pose.position.y = start_pose.pose.position.y + t * dy
            waypoint.pose.position.z = start_pose.pose.position.z  # Maintain height

            # Calculate orientation toward goal
            target_yaw = math.atan2(dy, dx)
            waypoint.pose.orientation = self.yaw_to_quaternion(target_yaw)

            waypoints.append(waypoint)

        # Add final goal
        final_waypoint = PoseStamped()
        final_waypoint.header = Header()
        final_waypoint.header.stamp = self.get_clock().now().to_msg()
        final_waypoint.header.frame_id = "map"
        final_waypoint.pose = goal_pose.pose
        waypoints.append(final_waypoint)

        return waypoints

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion
        """
        from geometry_msgs.msg import Quaternion
        import math

        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q

    def check_bipedal_navigation_feasibility(self, path):
        """
        Check if the path is feasible for bipedal navigation
        """
        if len(path) < 2:
            return False, "Path too short"

        # Check for obstacles and terrain constraints
        for i in range(len(path) - 1):
            current_pose = path[i].pose
            next_pose = path[i + 1].pose

            # Calculate distance between consecutive poses
            dx = next_pose.position.x - current_pose.position.x
            dy = next_pose.position.y - current_pose.position.y
            step_distance = math.sqrt(dx*dx + dy*dy)

            # Check if step is too large for bipedal robot
            if step_distance > 0.5:  # Max step distance for bipedal
                return False, f"Step too large at point {i}: {step_distance:.2f}m"

        return True, "Path is feasible for bipedal navigation"

    def navigate_with_bipedal_constraints(self, goal_pose):
        """
        Navigate to goal with bipedal-specific constraints
        """
        # First, check if navigation is feasible
        path = self.create_bipedal_safe_path(self.get_current_pose(), goal_pose)
        feasible, reason = self.check_bipedal_navigation_feasibility(path)

        if not feasible:
            self.get_logger().error(f"Navigation not feasible: {reason}")
            return False

        # Send navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose

        self.get_logger().info("Sending navigation goal with bipedal constraints...")

        # Wait for action server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Navigation action server not available")
            return False

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)
        return future

def main(args=None):
    rclpy.init(args=args)

    planner = HumanoidPathPlanner()

    # Example: Navigate to a specific pose
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.pose.position.x = 5.0
    goal_pose.pose.position.y = 3.0
    goal_pose.pose.position.z = 0.0
    goal_pose.pose.orientation.w = 1.0  # No rotation

    future = planner.navigate_with_bipedal_constraints(goal_pose)

    if future:
        rclpy.spin_until_future_complete(planner, future)

    planner.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Bipedal Navigation Simulation

Here's an example of simulating bipedal-safe navigation scenarios:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import math

class BipedalNavigationSimulator(Node):
    def __init__(self):
        super().__init__('bipedal_navigation_simulator')

        # Robot state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

        # Bipedal-specific parameters
        self.max_linear_vel = 0.3  # m/s for bipedal safety
        self.max_angular_vel = 0.3  # rad/s for balance
        self.bipedal_step_size = 0.2  # Max step size for bipedal
        self.balance_threshold = 0.1  # Balance maintenance threshold

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        # Timer for simulation updates
        self.timer = self.create_timer(0.1, self.update_simulation)  # 10 Hz

        self.get_logger().info("Bipedal Navigation Simulator initialized")

    def update_simulation(self):
        """
        Update the simulation state based on current commands
        """
        # Update robot position based on current velocities
        dt = 0.1  # Time step

        # Update position with forward kinematics
        self.current_x += self.linear_vel * math.cos(self.current_yaw) * dt
        self.current_y += self.linear_vel * math.sin(self.current_yaw) * dt
        self.current_yaw += self.angular_vel * dt

        # Normalize yaw to [-pi, pi]
        while self.current_yaw > math.pi:
            self.current_yaw -= 2 * math.pi
        while self.current_yaw < -math.pi:
            self.current_yaw += 2 * math.pi

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        odom_msg.pose.pose.position.x = self.current_x
        odom_msg.pose.pose.position.y = self.current_y
        odom_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        from geometry_msgs.msg import Quaternion
        q = self.yaw_to_quaternion(self.current_yaw)
        odom_msg.pose.pose.orientation = q

        # Set velocities
        odom_msg.twist.twist.linear.x = self.linear_vel
        odom_msg.twist.twist.angular.z = self.angular_vel

        self.odom_pub.publish(odom_msg)

        # Publish simulated laser scan
        self.publish_simulated_scan()

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw angle to quaternion
        """
        from geometry_msgs.msg import Quaternion
        q = Quaternion()
        q.w = math.cos(yaw / 2.0)
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        return q

    def publish_simulated_scan(self):
        """
        Publish simulated laser scan data
        """
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'

        # Laser scan parameters
        scan_msg.angle_min = -math.pi / 2  # -90 degrees
        scan_msg.angle_max = math.pi / 2   # 90 degrees
        scan_msg.angle_increment = math.pi / 180  # 1 degree increments
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Generate simulated ranges (with some obstacles)
        num_readings = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment) + 1
        ranges = []

        for i in range(num_readings):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment

            # Simulate some obstacles in the environment
            distance = scan_msg.range_max  # Default to max range

            # Add some simulated obstacles
            if abs(angle) < 0.2:  # Front of robot
                if 2.0 < self.current_x < 4.0 and abs(self.current_y - 1.0) < 0.5:
                    distance = 1.5  # Obstacle ahead

            ranges.append(distance)

        scan_msg.ranges = ranges
        scan_msg.intensities = [1.0] * len(ranges)

        self.scan_pub.publish(scan_msg)

    def set_navigation_command(self, linear, angular):
        """
        Set navigation command with bipedal constraints
        """
        # Apply bipedal-specific limits
        self.linear_vel = max(-self.max_linear_vel, min(linear, self.max_linear_vel))
        self.angular_vel = max(-self.max_angular_vel, min(angular, self.max_angular_vel))

        # Create and publish command
        cmd_msg = Twist()
        cmd_msg.linear.x = self.linear_vel
        cmd_msg.angular.z = self.angular_vel

        self.cmd_vel_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)

    simulator = BipedalNavigationSimulator()

    # Example: Move forward slowly (bipedal-safe speed)
    simulator.set_navigation_command(0.1, 0.0)  # 0.1 m/s forward

    print("Starting bipedal navigation simulation...")
    print("The robot will move forward at a safe speed for bipedal locomotion.")

    try:
        rclpy.spin(simulator)
    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        simulator.set_navigation_command(0.0, 0.0)  # Stop the robot
        simulator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Troubleshooting Guide

### Common Installation Issues

- **ROS 2 Version Compatibility**: Ensure Nav2 packages match your ROS 2 distribution
- **Dependency Conflicts**: Resolve package dependencies with apt
- **Navigation Configuration**: Verify all required configuration files are present

### Navigation Performance Issues

- **Path Planning Failures**: Check costmap configuration and obstacle detection
- **Bipedal Instability**: Reduce navigation speeds and adjust controller parameters
- **Goal Not Reached**: Increase tolerances and verify transform frames

### Simulation Problems

- **TF Issues**: Ensure all coordinate frames are properly published
- **Sensor Data**: Verify sensor topics are correctly configured
- **Control Frequency**: Adjust controller frequency for stability

## Hands-On Exercises

### Exercise 1: Nav2 Installation and Configuration

1. Install Nav2 packages for humanoid robot navigation
2. Configure Nav2 parameters for bipedal locomotion constraints
3. Set up costmaps and planners for humanoid-specific navigation
4. Verify the configuration with a simple navigation test

### Exercise 2: Bipedal Navigation Planning

1. Configure global and local planners for bipedal robots
2. Set appropriate velocity and acceleration limits for bipedal locomotion
3. Test path planning in various simulated environments
4. Evaluate the safety margins and balance constraints

### Exercise 3: Navigation Integration and Testing

1. Integrate Nav2 with Isaac ROS perception systems
2. Create a complete perception-navigation pipeline
3. Test the integrated system in simulation
4. Validate the navigation performance and safety

## Assessment Criteria

- Students can configure Nav2 for bipedal humanoid robot navigation
- Students can implement global and local planners with humanoid constraints
- Students can simulate safe navigation scenarios for bipedal robots
- Students can integrate Nav2 with Isaac ROS for complete solutions
