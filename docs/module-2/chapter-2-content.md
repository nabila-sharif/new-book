# Chapter 2: High-Fidelity Environments with Unity

## Introduction to Unity for Robotic Simulation

Unity has emerged as a powerful platform for creating high-fidelity visual environments for robotics simulation. Unlike traditional physics simulators, Unity excels at creating photorealistic environments with advanced rendering capabilities, making it ideal for testing perception algorithms, human-robot interaction scenarios, and visual-based navigation systems.

The key advantages of using Unity for robotic simulation include:

- Advanced rendering pipelines (Built-in, URP, HDRP)
- Photorealistic lighting and materials
- Extensive asset library and environment creation tools
- VR/AR support for immersive interaction
- Cross-platform deployment capabilities

## Creating High-Fidelity Environments for Humanoids

### Unity Rendering Pipelines

Unity offers three main rendering pipeline options:

#### Built-in Render Pipeline

- Default pipeline with basic lighting and shading
- Good performance for simple scenes
- Limited advanced rendering features
- Suitable for basic perception testing

#### Universal Render Pipeline (URP)

- Lightweight, optimized for performance
- Good balance between quality and performance
- Supports 2D and 3D rendering
- Ideal for real-time robotic applications

#### High Definition Render Pipeline (HDRP)

- Advanced rendering with physically-based lighting
- High-quality shadows, reflections, and post-processing
- Best for photorealistic environments
- Higher computational requirements

### Environment Setup Example

```csharp
using UnityEngine;
using UnityEngine.Rendering;

public class EnvironmentSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light sunLight;
    public Gradient skyGradient;

    [Header("Environment Materials")]
    public Material groundMaterial;
    public Material wallMaterial;

    void Start()
    {
        // Configure lighting
        ConfigureLighting();

        // Set up materials
        ConfigureMaterials();

        // Optimize for robotic simulation
        OptimizeForSimulation();
    }

    void ConfigureLighting()
    {
        // Set up realistic sun light
        sunLight.type = LightType.Directional;
        sunLight.intensity = 1.0f;
        sunLight.color = Color.white;

        // Configure shadows
        sunLight.shadows = LightShadows.Soft;
        sunLight.shadowStrength = 0.8f;
    }

    void ConfigureMaterials()
    {
        // Ground material with realistic properties
        groundMaterial.EnableKeyword("_METALLICGLOSSMAP");
        groundMaterial.SetFloat("_Metallic", 0.1f);
        groundMaterial.SetFloat("_Smoothness", 0.3f);
    }

    void OptimizeForSimulation()
    {
        // Reduce render quality for better performance
        QualitySettings.vSyncCount = 0;
        Application.targetFrameRate = 60;
    }
}
```

### Creating Realistic Indoor Environments

For humanoid robots, realistic indoor environments are crucial for testing navigation and interaction capabilities:

#### Room Layout Design

- Create accurate room dimensions based on real-world spaces
- Include furniture and obstacles that humanoid robots might encounter
- Ensure proper scale for humanoid proportions (typically 2-3 meters ceiling height)

#### Surface Materials

- Use PBR materials for realistic lighting response
- Configure friction properties for different surfaces
- Include texture variations for visual recognition

```csharp
public class SurfaceMaterialManager : MonoBehaviour
{
    public Material[] surfaceMaterials;

    [System.Serializable]
    public class SurfaceProperties
    {
        public string surfaceType;
        public float frictionCoefficient;
        public float restitution;
        public PhysicMaterial physicMaterial;
    }

    public SurfaceProperties[] surfaceProperties;

    void Start()
    {
        InitializeSurfaceMaterials();
    }

    void InitializeSurfaceMaterials()
    {
        foreach (var prop in surfaceProperties)
        {
            prop.physicMaterial = new PhysicMaterial();
            prop.physicMaterial.staticFriction = prop.frictionCoefficient;
            prop.physicMaterial.dynamicFriction = prop.frictionCoefficient;
            prop.physicMaterial.bounciness = prop.restitution;
        }
    }
}
```

### Outdoor Environment Considerations

For humanoid robots that operate outdoors, consider:

- Dynamic lighting conditions throughout the day
- Weather effects (rain, snow, fog)
- Terrain variations (grass, pavement, uneven surfaces)
- Natural obstacles (trees, rocks, water)

## Human-Robot Interaction Scenarios

### Interaction Design Principles

Creating effective human-robot interaction scenarios requires careful consideration of:

1. **Intuitive Control Interfaces**
2. **Natural Communication Methods**
3. **Safety Protocols**
4. **Feedback Systems**

### Implementing Interaction Scenarios

```csharp
using UnityEngine;
using System.Collections;

public class HumanRobotInteraction : MonoBehaviour
{
    [Header("Interaction Components")]
    public GameObject humanoidRobot;
    public Camera mainCamera;
    public LayerMask interactionMask;

    [Header("Interaction Parameters")]
    public float interactionDistance = 2.0f;
    public float interactionCooldown = 0.5f;

    private bool canInteract = true;
    private GameObject currentTarget;

    void Update()
    {
        HandleInteractionInput();
    }

    void HandleInteractionInput()
    {
        if (Input.GetMouseButtonDown(0) && canInteract)
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, interactionDistance, interactionMask))
            {
                currentTarget = hit.collider.gameObject;
                StartCoroutine(ProcessInteraction());
            }
        }
    }

    IEnumerator ProcessInteraction()
    {
        canInteract = false;

        // Process the interaction
        if (currentTarget.CompareTag("InteractiveObject"))
        {
            ProcessObjectInteraction(currentTarget);
        }
        else if (currentTarget.CompareTag("HumanoidRobot"))
        {
            ProcessRobotInteraction(currentTarget);
        }

        yield return new WaitForSeconds(interactionCooldown);
        canInteract = true;
    }

    void ProcessObjectInteraction(GameObject target)
    {
        // Handle object-specific interactions
        Debug.Log("Interacting with object: " + target.name);
    }

    void ProcessRobotInteraction(GameObject robot)
    {
        // Handle robot-specific interactions
        Debug.Log("Interacting with robot: " + robot.name);

        // Example: Send command to robot
        SendRobotCommand(robot, "GREET");
    }

    void SendRobotCommand(GameObject robot, string command)
    {
        // In a real implementation, this would communicate with ROS
        Debug.Log("Sending command to robot: " + command);
    }
}
```

### Safety Considerations in Interaction Design

- Implement collision avoidance systems
- Set safe interaction boundaries
- Include emergency stop functionality
- Provide clear visual feedback for robot state

## Unity-ROS Communication Workflow

### Setting up ROS-TCP-Connector

The ROS-TCP-Connector enables communication between Unity and ROS systems:

1. **Install ROS-TCP-Connector package**
2. **Configure network settings**
3. **Implement message serialization**
4. **Handle connection management**

### Basic ROS Connection Setup

```csharp
using UnityEngine;
using System.Collections;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

public class ROSConnectionManager : MonoBehaviour
{
    [Header("ROS Connection Settings")]
    public string rosIPAddress = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Robot Configuration")]
    public string robotName = "humanoid_robot";

    private ROSConnection rosConnection;

    void Start()
    {
        ConnectToROS();
    }

    void ConnectToROS()
    {
        rosConnection = ROSConnection.GetOrCreateInstance();
        rosConnection.rosIPAddress = rosIPAddress;
        rosConnection.rosPort = rosPort;

        // Register topics
        rosConnection.RegisterPublisher<sensor_msgs.JointState>("/joint_states");
        rosConnection.RegisterSubscriber<sensor_msgs.JointState>("/joint_states", OnJointStateReceived);

        Debug.Log("Connected to ROS at " + rosIPAddress + ":" + rosPort);
    }

    void OnJointStateReceived(sensor_msgs.JointState jointState)
    {
        // Process joint state data
        UpdateRobotJoints(jointState);
    }

    void UpdateRobotJoints(sensor_msgs.JointState jointState)
    {
        // Update Unity robot model based on ROS joint states
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = (float)jointState.position[i];

            Transform jointTransform = FindJointByName(jointName);
            if (jointTransform != null)
            {
                // Update joint rotation based on ROS data
                jointTransform.localRotation = Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
            }
        }
    }

    Transform FindJointByName(string jointName)
    {
        // Find joint transform in the robot hierarchy
        Transform[] allChildren = GetComponentsInChildren<Transform>();
        foreach (Transform child in allChildren)
        {
            if (child.name == jointName)
                return child;
        }
        return null;
    }

    void OnDestroy()
    {
        if (rosConnection != null)
        {
            rosConnection.Close();
        }
    }
}
```

### Sensor Data Integration

```csharp
using Unity.Robotics.ROSTCPConnector.MessageGeneration;

public class SensorDataHandler : MonoBehaviour
{
    [Header("Sensor Topics")]
    public string cameraTopic = "/camera/rgb/image_raw";
    public string lidarTopic = "/scan";
    public string imuTopic = "/imu/data";

    void Start()
    {
        // Subscribe to sensor topics
        ROSConnection.GetOrCreateInstance()
            .Subscribe<sensor_msgs.Image>(cameraTopic, OnCameraDataReceived);

        ROSConnection.GetOrCreateInstance()
            .Subscribe<sensor_msgs.LaserScan>(lidarTopic, OnLidarDataReceived);

        ROSConnection.GetOrCreateInstance()
            .Subscribe<sensor_msgs.Imu>(imuTopic, OnIMUDataReceived);
    }

    void OnCameraDataReceived(sensor_msgs.Image image)
    {
        // Process camera image data
        Texture2D cameraTexture = ConvertImageToTexture(image);
        UpdateCameraTexture(cameraTexture);
    }

    void OnLidarDataReceived(sensor_msgs.LaserScan scan)
    {
        // Process LIDAR scan data
        ProcessLidarScan(scan);
    }

    void OnIMUDataReceived(sensor_msgs.Imu imu)
    {
        // Process IMU data
        UpdateRobotOrientation(imu.orientation);
        UpdateRobotAcceleration(imu.linear_acceleration);
    }

    Texture2D ConvertImageToTexture(sensor_msgs.Image image)
    {
        // Convert ROS image message to Unity texture
        Texture2D texture = new Texture2D((int)image.width, (int)image.height, TextureFormat.RGB24, false);
        texture.LoadRawTextureData(image.data);
        texture.Apply();
        return texture;
    }

    void UpdateCameraTexture(Texture2D texture)
    {
        // Update material or UI element with camera texture
        Renderer renderer = GetComponent<Renderer>();
        if (renderer != null)
        {
            renderer.material.mainTexture = texture;
        }
    }

    void ProcessLidarScan(sensor_msgs.LaserScan scan)
    {
        // Process LIDAR data for visualization or navigation
        Debug.Log("Received LIDAR scan with " + scan.ranges.Count + " points");
    }

    void UpdateRobotOrientation(geometry_msgs.Quaternion orientation)
    {
        // Update robot orientation based on IMU data
        transform.rotation = new Quaternion(
            (float)orientation.x,
            (float)orientation.y,
            (float)orientation.z,
            (float)orientation.w
        );
    }

    void UpdateRobotAcceleration(geometry_msgs.Vector3 acceleration)
    {
        // Process linear acceleration data
        Debug.Log("Linear acceleration: " + acceleration);
    }
}
```

## Performance and Realism Trade-offs

### Balancing Visual Quality and Performance

For robotic simulation, finding the right balance between visual fidelity and performance is crucial:

#### Quality Settings for Robotics

```csharp
public class RoboticsQualitySettings : MonoBehaviour
{
    public enum QualityLevel
    {
        Performance,    // Lower quality, higher frame rate
        Balanced,       // Medium quality, good performance
        Quality         // High quality, lower frame rate
    }

    public QualityLevel currentQuality = QualityLevel.Balanced;

    void Start()
    {
        ApplyQualitySettings();
    }

    void ApplyQualitySettings()
    {
        switch (currentQuality)
        {
            case QualityLevel.Performance:
                QualitySettings.SetQualityLevel(0);
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = 60;
                break;

            case QualityLevel.Balanced:
                QualitySettings.SetQualityLevel(1);
                QualitySettings.vSyncCount = 1;
                Application.targetFrameRate = 30;
                break;

            case QualityLevel.Quality:
                QualitySettings.SetQualityLevel(2);
                QualitySettings.vSyncCount = 1;
                Application.targetFrameRate = 30;
                break;
        }
    }
}
```

#### Optimization Techniques

1. **Level of Detail (LOD) Systems**
2. **Occlusion Culling**
3. **Texture Streaming**
4. **Dynamic Batching**

### Realism vs. Performance Matrix

| Aspect | High Realism | Balanced | Performance |

|--------|--------------|----------|-------------|
| Lighting | HDRP, Real-time GI | URP, Baked lighting | Built-in, Simple lighting |
| Textures | 4K PBR materials | 2K textures | Compressed textures |
| Shadows | High-res, soft shadows | Medium-res shadows | Low-res shadows |
| Effects | Full post-processing | Limited effects | Minimal effects |
| Frame Rate | 30 FPS | 30-60 FPS | 60+ FPS |

## Best Practices for Unity-Based Robotic Simulation

### Asset Optimization

- Use appropriate polygon counts for real-time performance
- Implement texture atlasing for better draw call performance
- Use occlusion culling for complex environments
- Implement LOD systems for distant objects

### Simulation Accuracy

- Validate visual properties against real-world references
- Calibrate camera parameters to match physical sensors
- Ensure proper scale relationships between objects
- Test under various lighting conditions

### Integration Considerations

- Maintain consistent coordinate systems between Unity and ROS
- Implement proper time synchronization
- Handle network latency in real-time applications
- Include fallback mechanisms for connection failures

## Summary

This chapter covered the creation of high-fidelity environments using Unity for digital twin applications in humanoid robotics. We explored rendering pipelines, environment creation techniques, human-robot interaction scenarios, and Unity-ROS integration. The balance between visual realism and performance is crucial for effective robotic simulation, and proper integration with ROS enables comprehensive digital twin functionality.
