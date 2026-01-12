---
sidebar_position: 3
title: 'Chapter 3: Capstone Project - Autonomous Humanoid'
---

Chapter 3: Capstone Project - Autonomous Humanoid

## Learning Objectives

By the end of this chapter, you will be able to:

- Build an end-to-end Vision-Language-Action (VLA) pipeline in simulation
- Implement navigation with obstacle avoidance for humanoid robots
- Integrate object detection and manipulation capabilities
- Create a complete demo workflow: voice → plan → act → feedback
- Design ROS 2 action graph integration for complex tasks
- Set up and configure simulation environments for humanoid robotics

## Key Topics

### 1. End-to-End VLA Pipeline in Simulation

Creating a complete Vision-Language-Action pipeline involves integrating all components from voice input to robotic action execution in a simulated environment.

#### Complete VLA System Architecture

```python
#!/usr/bin/env python3
import rospy
import whisper
import openai
import json
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal
from actionlib_msgs.msg import GoalStatusArray
import cv2
from cv_bridge import CvBridge

class VLAPipeline:
    def __init__(self):
        rospy.init_node('vla_pipeline')

        # Initialize components
        self.whisper_model = whisper.load_model("base")
        openai.api_key = rospy.get_param('~openai_api_key', 'your-key-here')
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.voice_sub = rospy.Subscriber('/voice_input', String, self.voice_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        self.voice_cmd_pub = rospy.Publisher('/processed_voice_commands', String, queue_size=10)
        self.navigation_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/vla_status', String, queue_size=10)

        # State variables
        self.current_image = None
        self.laser_data = None
        self.robot_pose = None

        rospy.loginfo("VLA Pipeline initialized")

    def voice_callback(self, msg):
        """Process voice commands through the full VLA pipeline"""
        command = msg.data
        rospy.loginfo(f"Received voice command: {command}")

        # Update status
        status_msg = String()
        status_msg.data = f"Processing voice command: {command}"
        self.status_pub.publish(status_msg)

        # Plan based on voice command and current state
        plan = self.generate_plan_from_voice_and_state(command)

        # Execute the plan
        if plan:
            self.execute_plan(plan)

    def image_callback(self, msg):
        """Process camera images for vision component"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting image: {e}")

    def laser_callback(self, msg):
        """Process laser scan data for navigation"""
        self.laser_data = msg

    def generate_plan_from_voice_and_state(self, voice_command):
        """Generate a plan using LLM based on voice command and current state"""
        # Get current state information
        state_info = {
            "current_image_available": self.current_image is not None,
            "laser_data_available": self.laser_data is not None,
            "environment_objects": self.detect_objects_in_image() if self.current_image is not None else []
        }

        # Create prompt for LLM
        prompt = f"""
        You are a cognitive planner for a humanoid robot. The robot has received the voice command: "{voice_command}"

        Current state information:
        - Objects detected: {state_info['environment_objects']}
        - Camera data available: {state_info['current_image_available']}
        - Laser data available: {state_info['laser_data_available']}

        Generate a sequence of actions to fulfill the command. The available actions are:
        - move_to_location: Move to a specific location (x, y coordinates)
        - detect_object: Look for a specific object in the environment
        - approach_object: Move close to an object
        - grasp_object: Pick up an object
        - place_object: Place an object at a location
        - speak: Make the robot speak a message

        Return the plan as a JSON list of actions with parameters.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            plan_text = response.choices[0].message.content
            # Extract JSON from response
            start_idx = plan_text.find('[')
            end_idx = plan_text.rfind(']') + 1
            plan_json = plan_text[start_idx:end_idx]
            plan = json.loads(plan_json)
            return plan
        except Exception as e:
            rospy.logerr(f"Error generating plan: {e}")
            return []

    def detect_objects_in_image(self):
        """Simple object detection in the current image"""
        if self.current_image is None:
            return []

        # This is a simplified example - in practice you'd use a proper object detection model
        # For this example, we'll just return some placeholder objects
        # In a real implementation, you'd use YOLO, Detectron2, or similar
        height, width, _ = self.current_image.shape

        # Simulate object detection results
        objects = []
        if np.random.random() > 0.5:  # Randomly detect objects for simulation
            objects.append({
                "name": "red_cup",
                "confidence": 0.85,
                "bbox": [int(width*0.4), int(height*0.4), int(width*0.6), int(height*0.6)]
            })
        if np.random.random() > 0.7:
            objects.append({
                "name": "book",
                "confidence": 0.78,
                "bbox": [int(width*0.2), int(height*0.3), int(width*0.4), int(height*0.5)]
            })

        return objects

    def execute_plan(self, plan):
        """Execute a plan step by step"""
        rospy.loginfo(f"Executing plan with {len(plan)} steps")

        for i, action in enumerate(plan):
            rospy.loginfo(f"Executing step {i+1}/{len(plan)}: {action['action']}")

            success = self.execute_action(action)

            if not success:
                rospy.logerr(f"Action failed: {action}")
                # You could implement recovery strategies here
                break

        rospy.loginfo("Plan execution completed")

    def execute_action(self, action):
        """Execute a single action"""
        action_name = action.get('action', '')
        params = action.get('parameters', {})

        if action_name == 'move_to_location':
            return self.move_to_location(params)
        elif action_name == 'detect_object':
            return self.detect_object(params)
        elif action_name == 'approach_object':
            return self.approach_object(params)
        elif action_name == 'grasp_object':
            return self.grasp_object(params)
        elif action_name == 'place_object':
            return self.place_object(params)
        elif action_name == 'speak':
            return self.speak(params)
        else:
            rospy.logwarn(f"Unknown action: {action_name}")
            return False

    def move_to_location(self, params):
        """Move to a specific location"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        theta = params.get('theta', 0.0)

        goal = MoveBaseActionGoal()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.goal.target_pose.header.frame_id = "map"
        goal.goal.target_pose.pose.position.x = x
        goal.goal.target_pose.pose.position.y = y
        goal.goal.target_pose.pose.orientation.z = np.sin(theta / 2)
        goal.goal.target_pose.pose.orientation.w = np.cos(theta / 2)

        self.navigation_pub.publish(goal)
        rospy.loginfo(f"Moving to location: ({x}, {y}, {theta})")
        return True

    def detect_object(self, params):
        """Detect a specific object"""
        object_name = params.get('object_name', 'any')
        rospy.loginfo(f"Looking for object: {object_name}")

        # In a real system, this would trigger object detection
        # For simulation, we'll just return success
        return True

    def approach_object(self, params):
        """Approach an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Approaching object: {object_name}")

        # In a real system, this would navigate to the object
        # For simulation, we'll just return success
        return True

    def grasp_object(self, params):
        """Grasp an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Grasping object: {object_name}")

        # In a real system, this would trigger the gripper
        # For simulation, we'll just return success
        return True

    def place_object(self, params):
        """Place an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Placing object: {object_name}")

        # In a real system, this would release the gripper
        # For simulation, we'll just return success
        return True

    def speak(self, params):
        """Make the robot speak"""
        message = params.get('message', 'Hello')
        rospy.loginfo(f"Robot says: {message}")

        # In a real system, this would trigger text-to-speech
        # For simulation, we'll just log the message
        return True

    def run(self):
        """Run the VLA pipeline"""
        rospy.spin()
```

### 2. Navigation with Obstacle Avoidance

Implementing safe navigation in dynamic environments:

#### Advanced Navigation System

```python
class AdvancedNavigationSystem:
    def __init__(self):
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.laser_data = None
        self.current_pose = None
        self.path = []
        self.current_waypoint = 0

    def laser_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.laser_data = msg

    def odom_callback(self, msg):
        """Process odometry data for current pose"""
        self.current_pose = msg.pose.pose

    def check_for_obstacles(self, direction='forward', distance=1.0):
        """Check for obstacles in a specific direction"""
        if not self.laser_data:
            return False

        # Calculate angle range for the direction
        if direction == 'forward':
            start_angle = -15  # degrees
            end_angle = 15
        elif direction == 'left':
            start_angle = 75
            end_angle = 105
        elif direction == 'right':
            start_angle = -105
            end_angle = -75
        else:
            start_angle = -180
            end_angle = 180

        # Convert angles to laser indices
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment

        start_idx = int((np.radians(start_angle) - angle_min) / angle_increment)
        end_idx = int((np.radians(end_angle) - angle_min) / angle_increment)

        start_idx = max(0, start_idx)
        end_idx = min(len(self.laser_data.ranges), end_idx)

        # Check for obstacles within the distance
        for i in range(start_idx, end_idx):
            if self.laser_data.ranges[i] < distance and not np.isnan(self.laser_data.ranges[i]):
                return True

        return False

    def navigate_with_obstacle_avoidance(self, goal_x, goal_y):
        """Navigate to goal with obstacle avoidance"""
        rate = rospy.Rate(10)  # 10 Hz

        while not rospy.is_shutdown():
            if not self.current_pose:
                rate.sleep()
                continue

            # Calculate direction to goal
            current_x = self.current_pose.position.x
            current_y = self.current_pose.position.y

            dx = goal_x - current_x
            dy = goal_y - current_y
            distance_to_goal = np.sqrt(dx**2 + dy**2)

            # Check if we've reached the goal
            if distance_to_goal < 0.5:  # 0.5m tolerance
                cmd = Twist()
                self.cmd_vel_pub.publish(cmd)  # Stop
                rospy.loginfo("Reached goal")
                return True

            # Check for obstacles in the path
            if self.check_for_obstacles('forward', 1.0):
                # Implement obstacle avoidance behavior
                cmd = Twist()
                if self.check_for_obstacles('left', 1.0):
                    # Turn right if left is blocked
                    cmd.angular.z = -0.5
                else:
                    # Turn left to avoid obstacle
                    cmd.angular.z = 0.5
                cmd.linear.x = 0.1  # Slow forward movement
            else:
                # Move toward goal
                cmd = Twist()
                cmd.linear.x = min(0.5, distance_to_goal)  # Scale speed with distance
                cmd.angular.z = np.arctan2(dy, dx) - self.current_pose.orientation.z

            self.cmd_vel_pub.publish(cmd)
            rate.sleep()

        return False
```

### 3. Object Detection and Manipulation

Integrating vision and manipulation capabilities:

#### Vision-Based Manipulation System

```python
class VisionBasedManipulation:
    def __init__(self):
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)
        self.manipulation_pub = rospy.Publisher('/manipulation_commands', String, queue_size=10)

        self.cv_bridge = CvBridge()
        self.current_image = None
        self.current_depth = None
        self.detected_objects = []

    def image_callback(self, msg):
        """Process camera image for object detection"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detected_objects = self.detect_objects(self.current_image)
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def depth_callback(self, msg):
        """Process depth image for 3D information"""
        try:
            self.current_depth = self.cv_bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Error processing depth image: {e}")

    def detect_objects(self, image):
        """Detect objects in the image"""
        # This is a simplified example using color-based detection
        # In practice, you'd use a deep learning model like YOLO
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = {
            'red_cup': (np.array([0, 50, 50]), np.array([10, 255, 255])),
            'blue_bottle': (np.array([100, 50, 50]), np.array([130, 255, 255])),
            'green_box': (np.array([50, 50, 50]), np.array([70, 255, 255]))
        }

        detected = []
        for obj_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out small detections
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2

                    # Get depth at center of object
                    depth = self.get_depth_at_pixel(center_x, center_y) if self.current_depth is not None else None

                    detected.append({
                        'name': obj_name,
                        'bbox': [x, y, x+w, y+h],
                        'center': (center_x, center_y),
                        'depth': depth,
                        'confidence': 0.8  # Simplified confidence
                    })

        return detected

    def get_depth_at_pixel(self, x, y):
        """Get depth value at a specific pixel"""
        if self.current_depth is not None and 0 <= x < self.current_depth.shape[1] and 0 <= y < self.current_depth.shape[0]:
            return self.current_depth[y, x]
        return None

    def find_object_by_name(self, name):
        """Find a specific object by name"""
        for obj in self.detected_objects:
            if obj['name'] == name:
                return obj
        return None

    def approach_object(self, obj_name):
        """Approach a specific object"""
        obj = self.find_object_by_name(obj_name)
        if not obj:
            rospy.logerr(f"Object {obj_name} not found")
            return False

        # Calculate approach position (in front of object)
        if obj['depth']:
            approach_x = obj['center'][0]
            approach_y = obj['center'][1]
            distance = max(0.3, obj['depth'] - 0.2)  # 20cm from object

            # Send approach command
            cmd = String()
            cmd.data = f"approach_object:{obj_name}:{distance}"
            self.manipulation_pub.publish(cmd)
            return True
        else:
            rospy.logerr(f"No depth information for object {obj_name}")
            return False

    def grasp_object(self, obj_name):
        """Grasp a specific object"""
        obj = self.find_object_by_name(obj_name)
        if not obj:
            rospy.logerr(f"Object {obj_name} not found")
            return False

        # Send grasp command
        cmd = String()
        cmd.data = f"grasp_object:{obj_name}"
        self.manipulation_pub.publish(cmd)
        rospy.loginfo(f"Attempting to grasp {obj_name}")
        return True
```

### 4. Complete Demo Workflow: Voice → Plan → Act → Feedback

Creating a complete demonstration system:

#### Integrated Demo System

```python
class IntegratedDemoSystem:
    def __init__(self):
        rospy.init_node('integrated_demo_system')

        # Initialize all components
        self.vla_pipeline = VLAPipeline()
        self.nav_system = AdvancedNavigationSystem()
        self.vision_manip = VisionBasedManipulation()

        # Demo state
        self.demo_state = "idle"
        self.demo_steps = []
        self.current_step = 0

    def start_demo(self, voice_command):
        """Start the complete demo workflow"""
        rospy.loginfo(f"Starting demo with command: {voice_command}")

        # Step 1: Voice processing
        self.demo_state = "voice_processing"
        rospy.loginfo("Step 1: Processing voice command")

        # Generate plan from voice command
        plan = self.vla_pipeline.generate_plan_from_voice_and_state(voice_command)

        if not plan:
            rospy.logerr("Could not generate plan from voice command")
            return False

        rospy.loginfo(f"Generated plan with {len(plan)} steps")

        # Step 2: Plan execution
        self.demo_state = "planning"
        rospy.loginfo("Step 2: Executing plan")

        # Execute the plan
        success = self.vla_pipeline.execute_plan(plan)

        if success:
            # Step 3: Action execution with feedback
            self.demo_state = "acting"
            rospy.loginfo("Step 3: Actions completed, providing feedback")

            # Provide success feedback
            feedback_msg = String()
            feedback_msg.data = f"Successfully completed task: {voice_command}"
            self.vla_pipeline.status_pub.publish(feedback_msg)

            self.demo_state = "completed"
            rospy.loginfo("Demo completed successfully")
            return True
        else:
            rospy.logerr("Plan execution failed")
            feedback_msg = String()
            feedback_msg.data = f"Failed to complete task: {voice_command}"
            self.vla_pipeline.status_pub.publish(feedback_msg)

            self.demo_state = "failed"
            return False

    def run_predefined_demo(self):
        """Run a predefined demo sequence"""
        demo_commands = [
            "Go to the kitchen and bring me a red cup",
            "Navigate to the living room and find the book",
            "Move to the bedroom and turn left"
        ]

        for i, command in enumerate(demo_commands):
            rospy.loginfo(f"Running demo {i+1}/{len(demo_commands)}: {command}")

            success = self.start_demo(command)

            if not success:
                rospy.logerr(f"Demo {i+1} failed")
                break

            # Wait between demos
            rospy.sleep(5.0)

        rospy.loginfo("All demos completed")

    def run(self):
        """Run the integrated demo system"""
        rospy.loginfo("Integrated Demo System running")

        # Example: Start a predefined demo
        self.run_predefined_demo()

        rospy.spin()
```

### 5. ROS 2 Action Graph Integration

For ROS 2 systems, integrating action graphs:

#### ROS 2 Action Integration

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from example_interfaces.action import Fibonacci

class ROS2ActionIntegrator(Node):
    def __init__(self):
        super().__init__('ros2_action_integrator')

        # Create action clients
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Wait for action servers
        self.nav_client.wait_for_server()

    async def navigate_to_pose_async(self, x, y, theta):
        """Asynchronously navigate to a pose"""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2)

        self.get_logger().info(f'Waiting for action server to navigate to ({x}, {y}, {theta})')

        goal_handle = await self.nav_client.send_goal_async(goal_msg)

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return False

        self.get_logger().info('Goal accepted')

        result = await goal_handle.get_result_async()
        status = result.result
        self.get_logger().info(f'Result: {status}')

        return True

    def create_action_graph(self, plan):
        """Create an action graph from a plan"""
        # This would create a directed graph of actions with dependencies
        # For this example, we'll just return a sequence
        action_graph = {
            'nodes': [],
            'edges': []
        }

        for i, action in enumerate(plan):
            node = {
                'id': i,
                'action': action,
                'dependencies': [] if i == 0 else [i-1]  # Sequential dependencies
            }
            action_graph['nodes'].append(node)

            if i > 0:
                edge = {
                    'from': i-1,
                    'to': i,
                    'type': 'sequential'
                }
                action_graph['edges'].append(edge)

        return action_graph
```

### 6. Simulation Environment Setup

Setting up the simulation environment:

#### Simulation Configuration

```python
# simulation_setup.launch.py (for ROS 2)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='small_room.world',
        description='Choose one of the world files from `/gazebo_ros_pkgs/gazebo_worlds`'
    )

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': LaunchConfiguration('world')
        }.items()
    )

    # Launch robot model in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'humanoid_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.0'
        ],
        output='screen'
    )

    # Launch VLA pipeline
    vla_pipeline = Node(
        package='vla_package',
        executable='vla_pipeline',
        name='vla_pipeline',
        parameters=[
            {'openai_api_key': 'your-api-key-here'}
        ],
        output='screen'
    )

    # Launch navigation stack
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py'
            ])
        ])
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_entity,
        vla_pipeline,
        navigation
    ])
```

### 7. Complete Capstone Implementation

Here's the complete capstone project implementation:

```python
#!/usr/bin/env python3
import rospy
import whisper
import openai
import json
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionGoal
from cv_bridge import CvBridge

class AutonomousHumanoidCapstone:
    def __init__(self):
        rospy.init_node('autonomous_humanoid_capstone')

        # Initialize all components
        self.whisper_model = whisper.load_model("base")
        openai.api_key = rospy.get_param('~openai_api_key', 'your-api-key-here')
        self.cv_bridge = CvBridge()

        # Publishers and subscribers
        self.voice_sub = rospy.Subscriber('/voice_input', String, self.voice_command_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)

        self.voice_cmd_pub = rospy.Publisher('/processed_voice_commands', String, queue_size=10)
        self.navigation_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.status_pub = rospy.Publisher('/capstone_status', String, queue_size=10)
        self.manipulation_pub = rospy.Publisher('/manipulation_commands', String, queue_size=10)

        # State variables
        self.current_image = None
        self.laser_data = None
        self.system_state = "idle"
        self.detected_objects = []

        rospy.loginfo("Autonomous Humanoid Capstone System initialized")

    def voice_command_callback(self, msg):
        """Main callback for processing voice commands"""
        command = msg.data
        rospy.loginfo(f"Received voice command: {command}")

        # Update status
        status_msg = String()
        status_msg.data = f"Processing voice command: {command}"
        self.status_pub.publish(status_msg)

        # Execute the full VLA pipeline
        success = self.execute_vla_pipeline(command)

        if success:
            status_msg.data = f"Successfully completed: {command}"
        else:
            status_msg.data = f"Failed to complete: {command}"

        self.status_pub.publish(status_msg)

    def image_callback(self, msg):
        """Process camera images"""
        try:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            self.detected_objects = self.detect_objects_in_image()
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def laser_callback(self, msg):
        """Process laser scan data"""
        self.laser_data = msg

    def detect_objects_in_image(self):
        """Detect objects in the current image"""
        if self.current_image is None:
            return []

        # Simplified object detection (in practice, use a trained model)
        height, width, _ = self.current_image.shape

        objects = []
        # Simulate detecting some objects
        if np.random.random() > 0.3:
            objects.append({
                "name": "red_cup",
                "confidence": 0.85,
                "bbox": [int(width*0.4), int(height*0.4), int(width*0.6), int(height*0.6)],
                "center": (int(width*0.5), int(height*0.5))
            })
        if np.random.random() > 0.5:
            objects.append({
                "name": "book",
                "confidence": 0.78,
                "bbox": [int(width*0.2), int(height*0.3), int(width*0.4), int(height*0.5)],
                "center": (int(width*0.3), int(height*0.4))
            })

        return objects

    def generate_plan_with_llm(self, command, state_info):
        """Generate a plan using LLM based on command and state"""
        prompt = f"""
        You are a cognitive planner for an autonomous humanoid robot.
        The robot has received the command: "{command}"

        Current state information:
        - Objects detected: {[obj['name'] for obj in state_info['detected_objects']]}
        - Camera data available: {state_info['camera_available']}
        - Laser data available: {state_info['laser_available']}

        Generate a detailed sequence of actions to fulfill the command.
        Consider the detected objects and environmental constraints.

        Available actions:
        - navigate_to: Move to a location (x, y coordinates)
        - detect_object: Look for a specific object
        - approach_object: Move close to an object
        - grasp_object: Pick up an object
        - place_object: Place an object at a location
        - speak: Make the robot speak a message
        - wait: Pause for a specified time

        Return the plan as a JSON list of actions with parameters.
        Each action should have: action, parameters, and reasoning.
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            plan_text = response.choices[0].message.content
            # Extract JSON from response
            start_idx = plan_text.find('[')
            end_idx = plan_text.rfind(']') + 1
            plan_json = plan_text[start_idx:end_idx]
            plan = json.loads(plan_json)
            return plan
        except Exception as e:
            rospy.logerr(f"Error generating plan with LLM: {e}")
            return []

    def execute_vla_pipeline(self, command):
        """Execute the complete VLA pipeline: Voice → Language → Action"""
        rospy.loginfo("Starting VLA pipeline execution")

        # Gather current state
        state_info = {
            "detected_objects": self.detected_objects,
            "camera_available": self.current_image is not None,
            "laser_available": self.laser_data is not None
        }

        # Step 1: Plan generation using LLM
        rospy.loginfo("Step 1: Generating plan with LLM")
        plan = self.generate_plan_with_llm(command, state_info)

        if not plan:
            rospy.logerr("Could not generate plan")
            return False

        rospy.loginfo(f"Generated plan with {len(plan)} steps")

        # Step 2: Plan execution
        rospy.loginfo("Step 2: Executing plan")
        success = self.execute_plan(plan)

        if success:
            rospy.loginfo("VLA pipeline completed successfully")
            return True
        else:
            rospy.logerr("VLA pipeline execution failed")
            return False

    def execute_plan(self, plan):
        """Execute a plan step by step"""
        for i, action in enumerate(plan):
            rospy.loginfo(f"Executing action {i+1}/{len(plan)}: {action.get('action', 'unknown')}")

            success = self.execute_single_action(action)

            if not success:
                rospy.logerr(f"Action failed: {action}")
                return False

            # Small delay between actions
            rospy.sleep(0.5)

        return True

    def execute_single_action(self, action):
        """Execute a single action"""
        action_name = action.get('action', '')
        params = action.get('parameters', {})

        if action_name == 'navigate_to':
            return self.navigate_to_location(params)
        elif action_name == 'detect_object':
            return self.detect_object_action(params)
        elif action_name == 'approach_object':
            return self.approach_object_action(params)
        elif action_name == 'grasp_object':
            return self.grasp_object_action(params)
        elif action_name == 'place_object':
            return self.place_object_action(params)
        elif action_name == 'speak':
            return self.speak_action(params)
        elif action_name == 'wait':
            return self.wait_action(params)
        else:
            rospy.logwarn(f"Unknown action: {action_name}")
            return False

    def navigate_to_location(self, params):
        """Navigate to a specific location"""
        x = params.get('x', 0.0)
        y = params.get('y', 0.0)
        theta = params.get('theta', 0.0)

        goal = MoveBaseActionGoal()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.goal.target_pose.header.frame_id = "map"
        goal.goal.target_pose.pose.position.x = x
        goal.goal.target_pose.pose.position.y = y
        goal.goal.target_pose.pose.orientation.z = np.sin(theta / 2)
        goal.goal.target_pose.pose.orientation.w = np.cos(theta / 2)

        self.navigation_pub.publish(goal)
        rospy.loginfo(f"Navigating to ({x}, {y}, {theta})")
        rospy.sleep(2.0)  # Simulate navigation time
        return True

    def detect_object_action(self, params):
        """Detect a specific object"""
        object_name = params.get('object_name', 'any')
        rospy.loginfo(f"Detecting object: {object_name}")

        # In a real system, this would trigger object detection
        # For simulation, we'll just check if the object is in detected_objects
        for obj in self.detected_objects:
            if object_name.lower() in obj['name'].lower():
                rospy.loginfo(f"Found {obj['name']}")
                return True

        rospy.loginfo(f"Object {object_name} not found in current view")
        return True  # Not a failure, just not detected

    def approach_object_action(self, params):
        """Approach an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Approaching object: {object_name}")

        # In a real system, this would navigate to the object
        # For simulation, we'll just return success
        rospy.sleep(1.0)
        return True

    def grasp_object_action(self, params):
        """Grasp an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Grasping object: {object_name}")

        # In a real system, this would trigger the gripper
        # For simulation, we'll just return success
        rospy.sleep(1.0)
        return True

    def place_object_action(self, params):
        """Place an object"""
        object_name = params.get('object_name', 'unknown')
        rospy.loginfo(f"Placing object: {object_name}")

        # In a real system, this would release the gripper
        # For simulation, we'll just return success
        rospy.sleep(1.0)
        return True

    def speak_action(self, params):
        """Make the robot speak"""
        message = params.get('message', 'Hello')
        rospy.loginfo(f"Robot says: {message}")

        # In a real system, this would trigger text-to-speech
        # For simulation, we'll just log the message
        return True

    def wait_action(self, params):
        """Wait for a specified time"""
        duration = params.get('duration', 1.0)
        rospy.loginfo(f"Waiting for {duration} seconds")
        rospy.sleep(duration)
        return True

    def run(self):
        """Run the capstone system"""
        rospy.loginfo("Autonomous Humanoid Capstone System running")
        rospy.spin()

if __name__ == '__main__':
    capstone = AutonomousHumanoidCapstone()
    capstone.run()
```

## Assessment Criteria

- Students can build a complete end-to-end VLA pipeline in simulation
- Students can implement navigation with obstacle avoidance for humanoid robots
- Students can integrate object detection and manipulation capabilities
- Students can create a complete demo workflow from voice to action with feedback
- Students can design ROS 2 action graph integration for complex tasks
- Students can set up and configure simulation environments for humanoid robotics
- Students can demonstrate the complete autonomous humanoid system with voice commands
