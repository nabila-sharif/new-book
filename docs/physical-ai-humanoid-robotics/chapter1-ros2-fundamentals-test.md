---
sidebar_position: 5
---

# Chapter 1 Test: ROS 2 Fundamentals

## Learning Assessment

Complete the following exercises to verify your understanding of ROS 2 fundamentals.

### Question 1: Role of ROS 2 in Physical AI

Explain the role of ROS 2 in physical AI systems. How does it facilitate communication between different robot components?

### Question 2: Core Concepts

Identify and describe the following ROS 2 concepts:

- Nodes
- Topics and Messages
- Services
- Actions

### Question 3: Communication Patterns

Compare and contrast the different communication patterns in ROS 2:

- Publish/Subscribe (Topics)

- Request/Reply (Services)
- Goal/Feedback/Result (Actions)

### Practical Exercise

Create a simple ROS 2 system with:

1. A publisher node that sends sensor data
2. A subscriber node that receives and processes the data
3. A service node that responds to requests

## Answer Key

### Question 1 Answer

ROS 2 serves as the nervous system of a robot, providing communication infrastructure that enables different robot components to exchange information. It allows for modularity, distributed computing, and standardization of interfaces between robot components.

### Question 2 Answer

- **Nodes**: Fundamental units of execution that perform specific functions
- **Topics and Messages**: Publish/subscribe communication mechanism with structured data
- **Services**: Request/reply communication pattern for synchronous operations
- **Actions**: Extended services for long-running tasks with feedback

### Question 3 Answer

- Topics: One-way, asynchronous communication via publish/subscribe
- Services: Two-way, synchronous communication for immediate requests
- Actions: Two-way communication for long-running tasks with progress feedback
