---
sidebar_position: 6
---

# Chapter 2 Test: Python-based AI Agent

## Learning Assessment

Complete the following exercises to verify your understanding of Python-based AI agents and rclpy.

### Question 1: rclpy Implementation

Implement a Python-based ROS 2 node that demonstrates the use of publishers, subscribers, and services using rclpy. Your implementation should include:

- A subscriber to receive sensor data
- A publisher to send control commands
- A service to respond to external requests

### Question 2: AI Decision-Making

Describe how an AI algorithm can process sensor data and translate decisions into robot actions through ROS 2. Include in your explanation:

- The flow of data from sensors to actuators
- How state is maintained between decisions
- How machine learning models can be integrated

### Question 3: Communication Patterns

Compare the effectiveness of different communication patterns for AI agents:

- When to use topics (publish/subscribe)
- When to use services (request/reply)
- When to use actions (goal/feedback/result)

## Practical Exercise

Create a complete AI agent that:

1. Subscribes to sensor data (e.g., laser scan, camera)
2. Processes the data using a simple AI algorithm
3. Publishes control commands to move a robot
4. Responds to service requests for status information

## Answer Key

### Question 1 Answer

A proper implementation should include:

- Proper node initialization with rclpy
- Correct subscription setup with callback functions
- Proper publisher creation and message publishing
- Service server implementation with request handling

### Question 2 Answer

The flow involves: sensor data → AI processing → decision making → control commands. State can be maintained using class variables, and ML models can be integrated by loading models and using them in the decision-making process.

### Question 3 Answer

Topics for continuous data streams, services for discrete requests, and actions for long-running tasks with feedback.
