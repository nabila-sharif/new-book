---
sidebar_position: 3
---

# Chapter 2: Python-based AI Agent - From Decision to Action

This chapter covers using ROS 2 communication with Python using rclpy to create AI agents that can translate decisions into robot actions. We'll explore how to implement publishers, subscribers, and services to create effective AI agents that can interact with the physical world.

## Python and ROS 2 Integration

Python is one of the primary languages used in ROS 2 development. The rclpy package provides a Python client library for ROS 2, making it accessible for AI researchers and developers who prefer Python for its rich ecosystem of machine learning and AI libraries.

### Installing rclpy

rclpy is typically installed as part of a ROS 2 distribution. If you're setting up your environment, make sure to source your ROS 2 installation:

```bash
source /opt/ros/humble/setup.bash  # or your ROS 2 distribution
```

## Creating AI Agents with rclpy

An AI agent in the ROS 2 context is a node that receives sensor data, processes it using AI algorithms, and publishes control commands to actuators. This creates a complete decision-making loop that connects AI reasoning with physical action.

### Basic AI Agent Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np

class AIAgent(Node):
    def __init__(self):
        super().__init__('ai_agent')

        # Subscribe to sensor data
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.sensor_callback,
            10)

        # Publish control commands
        self.publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        # Timer for AI decision-making
        self.timer = self.create_timer(0.1, self.ai_decision_loop)

        self.latest_sensor_data = None
        self.get_logger().info('AI Agent initialized')

    def sensor_callback(self, msg):
        # Store latest sensor data for AI processing
        self.latest_sensor_data = msg.ranges

    def ai_decision_loop(self):
        if self.latest_sensor_data is not None:
            # Apply AI algorithm to sensor data
            control_command = self.make_decision(self.latest_sensor_data)

            # Publish the control command
            self.publisher.publish(control_command)

    def make_decision(self, sensor_data):
        # Simple AI algorithm: avoid obstacles
        msg = Twist()

        if min(sensor_data) > 1.0:  # No obstacles nearby
            msg.linear.x = 0.5  # Move forward
            msg.angular.z = 0.0
        else:  # Obstacle detected
            msg.linear.x = 0.0
            msg.angular.z = 0.5  # Turn right

        return msg

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AIAgent()
    rclpy.spin(ai_agent)
    ai_agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Communication Patterns with rclpy

### Publishers

Publishers send data to topics. In an AI agent context, publishers are used to send control commands to robot actuators.

```python
# Creating a publisher
self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

# Publishing a message
msg = Twist()
msg.linear.x = 0.5
msg.angular.z = 0.1
self.cmd_publisher.publish(msg)
```

### Subscribers

Subscribers receive data from topics. In an AI agent context, subscribers are used to receive sensor data from robot sensors.

```python
# Creating a subscriber
self.sensor_subscriber = self.create_subscription(
    LaserScan,
    'scan',
    self.process_sensor_data,
    10)

# Processing received data
def process_sensor_data(self, msg):
    # Process sensor data with AI algorithm
    self.handle_sensor_input(msg)
```

### Services

Services provide request/reply communication. In an AI agent context, services can be used for high-level commands or to request information.

```python
# Creating a service server
self.service = self.create_service(
    Trigger,
    'ai_decision_service',
    self.handle_decision_request)

# Handling service requests
def handle_decision_request(self, request, response):
    if self.ai_model.is_ready():
        response.success = True
        response.message = "AI decision ready"
    else:
        response.success = False
        response.message = "AI model not ready"
    return response
```

## Advanced AI Agent Patterns

### State Management

AI agents often need to maintain state between decisions. Here's how to implement state management in your agent:

```python
class StatefulAIAgent(Node):
    def __init__(self):
        super().__init__('stateful_ai_agent')
        self.agent_state = 'SEARCHING'  # Possible states: SEARCHING, APPROACHING, AVOIDING
        self.target_location = None
        self.memory = []  # Store past decisions

        # Setup subscribers, publishers, etc.

    def update_state(self, sensor_data):
        # Update agent state based on sensor data and current state
        if self.agent_state == 'SEARCHING' and self.detect_target(sensor_data):
            self.agent_state = 'APPROACHING'
            self.get_logger().info('State changed to APPROACHING')
```

### Integration with Machine Learning Models

AI agents often use pre-trained machine learning models for decision-making:

```python
import tensorflow as tf  # or your preferred ML framework

class MLIARobot(Node):
    def __init__(self):
        super().__init__('ml_ai_robot')

        # Load pre-trained model
        self.ml_model = tf.keras.models.load_model('path/to/model')

        # Setup ROS 2 communication
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)

    def image_callback(self, msg):
        # Convert ROS image to format suitable for ML model
        image_data = self.ros_image_to_ml_format(msg)

        # Get prediction from ML model
        prediction = self.ml_model.predict(image_data)

        # Convert prediction to robot action
        action = self.prediction_to_action(prediction)

        # Publish action
        self.publish_action(action)
```

## Complete Example: AI Decision to Robot Action Flow

Let's look at a complete example that demonstrates the full flow from AI decision-making to robot action. This example implements a simple object recognition and navigation system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectNavigationAgent(Node):
    def __init__(self):
        super().__init__('object_navigation_agent')

        # Initialize CV bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to camera and laser scanner
        self.image_subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10)

        self.scan_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            10)

        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publisher for object detection status
        self.status_publisher = self.create_publisher(String, 'object_status', 10)

        # Timer for decision-making loop
        self.timer = self.create_timer(0.1, self.decision_loop)

        # State variables
        self.latest_image = None
        self.latest_scan = None
        self.detected_object = None
        self.navigation_state = 'SEARCHING'  # SEARCHING, NAVIGATING, AVOIDING

        self.get_logger().info('Object Navigation Agent initialized')

    def image_callback(self, msg):
        """Process camera images to detect objects"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Simple color-based object detection (red object)
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # Define range for red color
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)

            lower_red = np.array([170, 50, 50])
            upper_red = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red, upper_red)

            mask = mask1 + mask2

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                    # Calculate center of the object
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # Normalize position (0 = left, 1 = right)
                        img_width = cv_image.shape[1]
                        object_position = cx / img_width

                        self.detected_object = {
                            'position': object_position,
                            'area': cv2.contourArea(largest_contour)
                        }

                        # Update navigation state
                        self.navigation_state = 'NAVIGATING'

                        # Publish object status
                        status_msg = String()
                        status_msg.data = f"Red object detected at position {object_position:.2f}"
                        self.status_publisher.publish(status_msg)
            else:
                self.detected_object = None
                self.navigation_state = 'SEARCHING'

            self.latest_image = cv_image

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def scan_callback(self, msg):
        """Process laser scan data for obstacle detection"""
        self.latest_scan = msg.ranges

    def decision_loop(self):
        """Main AI decision-making loop"""
        if self.navigation_state == 'SEARCHING':
            self.search_behavior()
        elif self.navigation_state == 'NAVIGATING':
            if self.detected_object:
                self.navigate_to_object()
            else:
                self.navigation_state = 'SEARCHING'
        elif self.navigation_state == 'AVOIDING':
            self.avoid_obstacles()

    def search_behavior(self):
        """Behavior when searching for objects"""
        cmd = Twist()
        cmd.linear.x = 0.2  # Move forward slowly
        cmd.angular.z = 0.3  # Rotate slowly to scan environment
        self.cmd_publisher.publish(cmd)

    def navigate_to_object(self):
        """Navigate toward the detected object"""
        if not self.detected_object or not self.latest_scan:
            return

        cmd = Twist()

        # Check for obstacles
        if self.latest_scan:
            min_distance = min([d for d in self.latest_scan if not np.isnan(d) and d > 0.1])
            if min_distance < 0.5:  # Obstacle too close
                self.navigation_state = 'AVOIDING'
                cmd.linear.x = 0.0
                cmd.angular.z = 0.5  # Turn away from obstacle
                self.cmd_publisher.publish(cmd)
                return

        # Navigate toward object
        object_pos = self.detected_object['position']

        if object_pos < 0.4:  # Object is to the left
            cmd.angular.z = 0.3
        elif object_pos > 0.6:  # Object is to the right
            cmd.angular.z = -0.3
        else:  # Object is centered
            cmd.linear.x = 0.3  # Move forward toward object

        self.cmd_publisher.publish(cmd)

    def avoid_obstacles(self):
        """Simple obstacle avoidance behavior"""
        if not self.latest_scan:
            return

        cmd = Twist()

        # Simple wall-following behavior
        left_scan = self.latest_scan[0:90]  # Left quarter
        front_scan = self.latest_scan[90:270]  # Front half
        right_scan = self.latest_scan[270:360]  # Right quarter

        min_front = min([d for d in front_scan if not np.isnan(d) and d > 0.1])
        min_left = min([d for d in left_scan if not np.isnan(d) and d > 0.1])
        min_right = min([d for d in right_scan if not np.isnan(d) and d > 0.1])

        if min_front > 0.8:  # Path is clear
            cmd.linear.x = 0.3
            cmd.angular.z = 0.0
            self.navigation_state = 'NAVIGATING'  # Return to navigation
        elif min_left > min_right:  # More space on left
            cmd.angular.z = 0.5
        else:  # More space on right
            cmd.angular.z = -0.5

        self.cmd_publisher.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    agent = ObjectNavigationAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Hands-on Exercise: Creating Your Own Python-based AI Agent

Now it's time to create your own AI agent. Follow these steps to build a complete system:

### Exercise 1: Basic AI Agent

1. Create a new Python file called `my_ai_agent.py`
2. Implement a basic AI agent that subscribes to sensor data
3. Implement a simple decision-making algorithm
4. Publish commands to control the robot

### Exercise 2: Enhanced Decision Making

1. Modify your agent to handle multiple sensor inputs
2. Implement state management to remember past decisions
3. Add a service that allows external systems to query the agent's state

### Exercise 3: Integration with Machine Learning

1. Integrate a simple machine learning model (you can use scikit-learn or TensorFlow)
2. Train the model to recognize patterns in sensor data
3. Use the model's predictions to influence the robot's behavior

### Solution Template

Here's a template to get you started:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan  # Add other message types as needed
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class MyAIAgent(Node):
    def __init__(self):
        super().__init__('my_ai_agent')

        # TODO: Setup your subscribers, publishers, and timers
        # TODO: Initialize your AI model or decision variables

    def sensor_callback(self, msg):
        # TODO: Process sensor data
        pass

    def ai_decision_callback(self):
        # TODO: Implement your AI decision-making logic
        pass

def main(args=None):
    rclpy.init(args=args)
    agent = MyAIAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Summary

In this chapter, we've explored how to create Python-based AI agents using rclpy and ROS 2. You've learned about:

- How to implement publishers, subscribers, and services using rclpy
- The complete flow from AI decision-making to robot action
- Practical examples of AI agents that process sensor data and control robot behavior
- Hands-on exercises to build your own AI agents

These skills enable you to create sophisticated AI systems that can interact with the physical world through robots. The next chapter will cover how to model robot structure using URDF, which is essential for simulation and control.

This chapter demonstrates the complete flow from AI decision-making to robot action, showing how Python and rclpy enable the creation of sophisticated AI agents that can process information and control robot behavior effectively.
