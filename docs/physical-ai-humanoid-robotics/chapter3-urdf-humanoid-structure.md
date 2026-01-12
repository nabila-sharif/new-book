---
sidebar_position: 4
---

# Chapter 3: Humanoid Robot Modeler - Defining Structure with URDF

This chapter covers defining humanoid robot structure using URDF (Unified Robot Description Format), understanding links, joints, and kinematic chains. URDF is the standard way to describe robot models in ROS and is essential for simulation, visualization, and control.

## What is URDF?

URDF (Unified Robot Description Format) is an XML format that describes robots. It's used to define the physical structure of a robot including links, joints, and other properties. URDF files are fundamental to robotics development as they allow:

- **Simulation**: Creating accurate physics models for testing
- **Visualization**: Displaying robot models in RViz and other tools
- **Control**: Understanding the kinematic structure for motion planning
- **Collision Detection**: Defining shapes for collision avoidance

### Basic URDF Structure

A basic URDF file has the following structure:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints define connections between links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
  </link>
</robot>
```

## Links: The Building Blocks of Robots

Links represent rigid bodies in a robot. They define the physical properties of robot parts including:

### Visual Properties

- **Geometry**: Shape (box, cylinder, sphere, mesh)
- **Material**: Color, texture, and visual appearance
- **Origin**: Position and orientation relative to joint

### Collision Properties

- **Collision geometry**: Used for physics simulation and collision detection
- **Shape**: Often simplified compared to visual geometry for performance

### Inertial Properties

- **Mass**: Physical mass of the link
- **Inertia**: How mass is distributed (important for dynamics)

### Example: Link Definition

```xml
<link name="upper_arm">
  <visual>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
  </inertial>
</link>
```

## Joints: Connecting the Parts

Joints define the relationship between links and specify how they can move relative to each other. There are several joint types:

### Joint Types

- **fixed**: No movement (welded connection)
- **continuous**: Rotation around axis (like a wheel)
- **revolute**: Limited rotation around axis (like elbow)
- **prismatic**: Linear movement along axis (like a slider)
- **floating**: 6DOF movement (not commonly used)

### Joint Properties

- **Parent/Child**: Links connected by the joint
- **Origin**: Position and orientation of the joint
- **Axis**: Direction of movement for revolute/prismatic joints
- **Limits**: Movement constraints for revolute and prismatic joints

### Example: Joint Definition

```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="1.5" effort="30" velocity="1.0"/>
</joint>
```

## Kinematic Chains: The Mathematical Foundation

Kinematic chains describe the mathematical relationships between different parts of a robot, allowing for:

- **Forward Kinematics**: Calculating end-effector position from joint angles
- **Inverse Kinematics**: Calculating joint angles to achieve desired end-effector position
- **Trajectory Planning**: Planning smooth movements through joint space

### Humanoid Kinematic Structure

A humanoid robot typically has multiple kinematic chains:

- Left arm chain (from torso to left hand)
- Right arm chain (from torso to right hand)
- Left leg chain (from torso to left foot)
- Right leg chain (from torso to right foot)

## Humanoid Robot URDF Example

Here's a complete example of a simplified humanoid robot:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Torso -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.6" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Left Arm -->
  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="1.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_forearm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="0.5" effort="30" velocity="1.0"/>
  </joint>

  <link name="left_forearm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>
</robot>
```

## Simulation and Control with URDF

URDF models are essential for both simulation and real-world robot control. Here's how they enable these applications:

### Gazebo Simulation

Gazebo is a popular 3D simulation environment that uses URDF models to create realistic robot simulations. To prepare a URDF for simulation, you need to add additional elements:

```xml
<!-- Transmission elements for simulation -->
<transmission name="left_elbow_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Gazebo-specific elements -->
<gazebo reference="left_upper_arm">
  <material>Gazebo/Blue</material>
</gazebo>

<gazebo reference="left_elbow_joint">
  <implicitSpringDamper>1</implicitSpringDamper>
</gazebo>
```

### ROS Control Integration

For real robot control, URDF models need to interface with ros_control, which provides a standard interface for controlling robot hardware:

```xml
<!-- ros_control plugin for simulation -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/simple_humanoid</robotNamespace>
  </plugin>
</gazebo>
```

## Practical URDF Creation Workflow

Here's a step-by-step workflow for creating effective URDF models:

### Step 1: Plan Your Robot Structure

- Sketch your robot design
- Identify all links and joints
- Determine joint types and limits
- Consider degrees of freedom needed

### Step 2: Create the Base Link

- Start with the root link (usually the main body)
- Define basic geometry and mass properties
- Set up visual and collision elements

### Step 3: Add Connected Links

- Define joints connecting new links to existing ones
- Ensure proper parent-child relationships
- Set appropriate origins and axes

### Step 4: Refine Physical Properties

- Accurate mass and inertia values
- Proper collision geometry
- Realistic joint limits

### Step 5: Test and Validate

- Load in RViz to check visualization
- Simulate in Gazebo to test physics
- Validate kinematic chain with forward/inverse kinematics

## Hands-on Exercise: Create Your Own URDF Model

Create a URDF model for a simple robot of your choice. Follow these steps:

### Exercise 1: Simple Mobile Base

1. Create a URDF file for a simple wheeled robot with:
   - A base link
   - Two wheel links connected with continuous joints
   - Proper visual and collision properties

### Exercise 2: Articulated Arm

1. Extend your model to include a simple 2-DOF arm
2. Add appropriate joints and links for shoulder and elbow
3. Ensure the kinematic chain is properly defined

### Exercise 3: Complete Humanoid Limb

1. Create a more complex model of a humanoid leg with hip, knee, and ankle joints
2. Include realistic joint limits based on human anatomy
3. Add proper mass and inertia properties

## Best Practices for URDF Development

- **Start Simple**: Begin with basic shapes and add complexity gradually
- **Use Meshes Sparingly**: Complex meshes impact simulation performance
- **Verify Mass Properties**: Unrealistic masses cause simulation instability
- **Check Joint Limits**: Ensure limits are realistic for your hardware
- **Test Regularly**: Load your URDF frequently in RViz to catch errors early
- **Use Xacro**: For complex robots, use Xacro to avoid repetition

## URDF Validation Tools

Several ROS tools help validate URDF models:

```bash
# Check URDF syntax
check_urdf my_robot.urdf

# Display robot information
urdf_to_graphiz my_robot.urdf

# Test with robot state publisher
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat my_robot.urdf)
```

## Summary

In this chapter, we've covered the fundamentals of URDF and how to model humanoid robot structure. You've learned about:

- How to represent robot's physical structure with links and joints
- How URDF models properly represent robot's kinematic structure
- How to create URDF models with practical examples
- How URDF enables simulation and control applications

These skills are essential for creating accurate robot models for simulation, visualization, and control. With all three chapters completed, you now have a comprehensive understanding of the robotic nervous system (ROS 2), AI agents (Python/rclpy), and robot modeling (URDF) that form the foundation of physical AI and humanoid robotics.

This chapter provides the foundation for understanding how to model humanoid robots using URDF, which is essential for simulation and control applications.
