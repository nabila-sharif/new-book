# Chapter 3: Sensor Simulation & Fidelity

## Introduction to Sensor Simulation in Digital Twins

Sensor simulation is a critical component of digital twin systems for robotics, as it provides the virtual robot with environmental perception capabilities that mirror those of its physical counterpart. Accurate sensor simulation enables:

- Development and testing of perception algorithms in a safe environment
- Training of machine learning models with synthetic data
- Validation of navigation and control systems
- Evaluation of robot performance under various environmental conditions

The fidelity of sensor simulation directly impacts the transferability of algorithms from simulation to reality, making it essential to model sensor characteristics, noise, and limitations accurately.

## LiDAR Simulation and Noise Characteristics

### LiDAR Physics in Simulation

LiDAR (Light Detection and Ranging) sensors in simulation are typically implemented using raycasting techniques that mimic the behavior of real laser beams. The simulation must account for:

- **Ray casting**: Virtual laser beams projected from the sensor origin
- **Distance measurement**: Calculation of distances to nearest obstacles
- **Angular resolution**: Horizontal and vertical beam spacing
- **Range limitations**: Maximum and minimum detection distances

### LiDAR Simulation Implementation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class LIDARSimulator : MonoBehaviour
{
    [Header("LIDAR Configuration")]
    public int horizontalRays = 360;
    public int verticalRays = 16;
    public float minAngle = -30f;
    public float maxAngle = 15f;
    public float maxRange = 20.0f;
    public float minRange = 0.1f;
    public LayerMask detectionMask = -1;

    [Header("Noise Parameters")]
    public float distanceNoiseStdDev = 0.02f;  // 2cm standard deviation
    public float angularNoiseStdDev = 0.001f;  // Small angular error

    private List<float> scanData;
    private float[] verticalAngles;

    void Start()
    {
        InitializeLIDAR();
    }

    void InitializeLIDAR()
    {
        scanData = new List<float>();
        CalculateVerticalAngles();
    }

    void CalculateVerticalAngles()
    {
        verticalAngles = new float[verticalRays];
        float angleStep = (maxAngle - minAngle) / (verticalRays - 1);

        for (int i = 0; i < verticalRays; i++)
        {
            verticalAngles[i] = minAngle + (i * angleStep);
        }
    }

    public float[] GetLIDARScan()
    {
        scanData.Clear();

        for (int h = 0; h < horizontalRays; h++)
        {
            float horizontalAngle = (360.0f / horizontalRays) * h * Mathf.Deg2Rad;

            for (int v = 0; v < verticalRays; v++)
            {
                float verticalAngle = verticalAngles[v] * Mathf.Deg2Rad;

                // Calculate ray direction
                Vector3 rayDirection = CalculateRayDirection(horizontalAngle, verticalAngle);

                // Perform raycast
                RaycastHit hit;
                if (Physics.Raycast(transform.position, rayDirection, out hit, maxRange, detectionMask))
                {
                    float distance = hit.distance;

                    // Add noise to the measurement
                    distance = AddDistanceNoise(distance);

                    scanData.Add(distance);
                }
                else
                {
                    // No obstacle detected within range
                    scanData.Add(maxRange + 1.0f); // Indicate no return
                }
            }
        }

        return scanData.ToArray();
    }

    Vector3 CalculateRayDirection(float horizontalAngle, float verticalAngle)
    {
        // Calculate the ray direction based on horizontal and vertical angles
        Vector3 direction = new Vector3(
            Mathf.Cos(verticalAngle) * Mathf.Cos(horizontalAngle),
            Mathf.Cos(verticalAngle) * Mathf.Sin(horizontalAngle),
            Mathf.Sin(verticalAngle)
        );

        // Transform to world space based on sensor orientation
        return transform.TransformDirection(direction);
    }

    float AddDistanceNoise(float distance)
    {
        // Add Gaussian noise to distance measurement
        float noise = RandomGaussian() * distanceNoiseStdDev;
        float noisyDistance = distance + noise;

        // Ensure distance is within valid range
        return Mathf.Clamp(noisyDistance, minRange, maxRange + 1.0f);
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random number generation
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < Mathf.Epsilon) u1 = Mathf.Epsilon;

        float gaussian = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return gaussian;
    }
}
```

### Noise Modeling for LiDAR Sensors

Real LiDAR sensors exhibit various types of noise and systematic errors:

#### Statistical Noise Models

- **Gaussian noise**: Random measurement errors following a normal distribution
- **Intensity-based noise**: Distance measurement errors that vary with return intensity
- **Multi-path interference**: Errors due to multiple reflections

#### Systematic Errors

- **Zero-point calibration errors**: Constant offset in measurements
- **Scale factor errors**: Proportional errors across the measurement range
- **Angular misalignment**: Errors in beam pointing direction

### Advanced LiDAR Simulation Features

```csharp
public class AdvancedLIDARSimulator : LIDARSimulator
{
    [Header("Advanced Noise Models")]
    public AnimationCurve distanceNoiseCurve;  // Distance-dependent noise
    public float temperatureCoefficient = 0.001f;  // Temperature effect
    public float vibrationNoise = 0.01f;  // Vibration-induced errors

    [Header("Environmental Effects")]
    public float atmosphericAttenuation = 0.001f;  // Fog/rain effect
    public float dustAttenuation = 0.002f;  // Dust/particle effect

    float AddAdvancedDistanceNoise(float distance, float temperature, float vibration)
    {
        // Base noise from distance curve
        float distanceBasedNoise = distanceNoiseCurve.Evaluate(distance) * distanceNoiseStdDev;

        // Temperature effect
        float temperatureEffect = temperature * temperatureCoefficient;

        // Vibration effect
        float vibrationEffect = vibration * vibrationNoise;

        // Combine all noise sources
        float totalNoise = Mathf.Sqrt(
            Mathf.Pow(distanceBasedNoise, 2) +
            Mathf.Pow(temperatureEffect, 2) +
            Mathf.Pow(vibrationEffect, 2)
        );

        return distance + RandomGaussian() * totalNoise;
    }

    float ApplyEnvironmentalEffects(float distance, float atmosphericDensity, float particleDensity)
    {
        // Apply atmospheric attenuation
        float atmosphericFactor = Mathf.Exp(-atmosphericAttenuation * distance * atmosphericDensity);

        // Apply dust/particle attenuation
        float dustFactor = Mathf.Exp(-dustAttenuation * distance * particleDensity);

        return distance / (atmosphericFactor * dustFactor);
    }
}
```

## Depth Camera and RGB-D Simulation

### Depth Camera Fundamentals

Depth cameras in simulation must accurately model:

- **Pinhole camera model**: Geometric projection of 3D points to 2D image
- **Depth measurement**: Distance from camera to scene points
- **Noise characteristics**: Sensor-specific noise patterns
- **Resolution limitations**: Finite pixel resolution and quantization

### Depth Camera Implementation

```csharp
using UnityEngine;
using System.Collections;

public class DepthCameraSimulator : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int width = 640;
    public int height = 480;
    public float fov = 60f;
    public float nearClip = 0.1f;
    public float farClip = 10.0f;

    [Header("Noise Parameters")]
    public float depthNoiseStdDev = 0.01f;  // 1cm standard deviation
    public float radialDistortion = 0.1f;   // Lens distortion
    public float tangentialDistortion = 0.01f;

    private Camera depthCamera;
    private RenderTexture depthTexture;
    private float[,] depthData;

    void Start()
    {
        SetupDepthCamera();
        CreateDepthTexture();
        depthData = new float[width, height];
    }

    void SetupDepthCamera()
    {
        depthCamera = GetComponent<Camera>();
        if (depthCamera == null)
        {
            depthCamera = gameObject.AddComponent<Camera>();
        }

        depthCamera.fieldOfView = fov;
        depthCamera.nearClipPlane = nearClip;
        depthCamera.farClipPlane = farClip;
        depthCamera.depth = -1; // Render after other cameras
        depthCamera.enabled = false; // Don't render automatically
    }

    void CreateDepthTexture()
    {
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        depthTexture.Create();
        depthCamera.targetTexture = depthTexture;
    }

    public float[,] GetDepthImage()
    {
        // Render the scene from depth camera
        depthCamera.Render();

        // Read depth data from texture
        RenderTexture.active = depthTexture;
        Texture2D depthTex = new Texture2D(width, height, TextureFormat.RGB24, false);
        depthTex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        depthTex.Apply();

        // Convert texture to depth values
        Color[] pixels = depthTex.GetPixels();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIndex = y * width + x;
                Color pixel = pixels[pixelIndex];

                // Convert color to depth value (simplified)
                float rawDepth = pixel.r; // Assuming red channel contains depth info
                float depth = ConvertRawDepthToMeters(rawDepth);

                // Add noise to depth measurement
                depth = AddDepthNoise(depth, x, y);

                depthData[x, y] = depth;
            }
        }

        // Clean up
        RenderTexture.active = null;
        DestroyImmediate(depthTex);

        return depthData;
    }

    float ConvertRawDepthToMeters(float rawDepth)
    {
        // Convert normalized depth value to meters
        // This depends on your specific depth rendering setup
        float depth = nearClip + rawDepth * (farClip - nearClip);
        return depth;
    }

    float AddDepthNoise(float depth, int x, int y)
    {
        // Add position-dependent noise
        float noise = RandomGaussian() * depthNoiseStdDev;

        // Add quantization noise based on pixel position
        float quantization = (x % 4) * 0.001f + (y % 4) * 0.001f;

        return depth + noise + quantization;
    }

    float RandomGaussian()
    {
        // Box-Muller transform
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < Mathf.Epsilon) u1 = Mathf.Epsilon;

        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }

    void OnDestroy()
    {
        if (depthTexture != null)
        {
            depthTexture.Release();
        }
    }
}
```

### RGB-D Sensor Pipeline

```csharp
public class RGBDSensorPipeline : MonoBehaviour
{
    [Header("RGB-D Configuration")]
    public DepthCameraSimulator depthCamera;
    public Camera rgbCamera;
    public int width = 640;
    public int height = 480;

    [Header("Calibration Parameters")]
    public Vector2 principalPoint = new Vector2(320, 240);
    public Vector2 focalLength = new Vector2(525, 525);
    public float baseline = 0.075f; // For stereo cameras

    private float[,] depthData;
    private Color32[] rgbData;
    private Vector3[,] pointCloud;

    void Start()
    {
        InitializeRGBDPipeline();
    }

    void InitializeRGBDPipeline()
    {
        // Initialize depth camera
        if (depthCamera == null)
        {
            depthCamera = GetComponent<DepthCameraSimulator>();
        }

        // Initialize RGB camera
        if (rgbCamera == null)
        {
            rgbCamera = GetComponent<Camera>();
        }

        pointCloud = new Vector3[width, height];
    }

    public void UpdateSensorData()
    {
        // Get depth and RGB data
        depthData = depthCamera.GetDepthImage();
        rgbData = GetRGBData();

        // Generate point cloud from depth data
        GeneratePointCloud();
    }

    Color32[] GetRGBData()
    {
        // Capture RGB image from camera
        RenderTexture currentRT = RenderTexture.active;
        RenderTexture.active = rgbCamera.targetTexture;

        Texture2D tex = new Texture2D(width, height, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        tex.Apply();

        Color32[] pixels = tex.GetPixels32();
        RenderTexture.active = currentRT;
        DestroyImmediate(tex);

        return pixels;
    }

    void GeneratePointCloud()
    {
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthData[x, y];

                if (depth > 0 && depth < depthCamera.farClip)
                {
                    // Convert pixel coordinates to 3D world coordinates
                    float worldX = (x - principalPoint.x) * depth / focalLength.x;
                    float worldY = (y - principalPoint.y) * depth / focalLength.y;
                    float worldZ = depth;

                    // Transform from camera to world coordinates
                    Vector3 localPoint = new Vector3(worldX, worldY, worldZ);
                    Vector3 worldPoint = transform.TransformPoint(localPoint);

                    pointCloud[x, y] = worldPoint;
                }
                else
                {
                    pointCloud[x, y] = Vector3.zero; // Invalid point
                }
            }
        }
    }

    public Vector3[,] GetPointCloud()
    {
        return pointCloud;
    }
}
```

## IMU Modeling and Drift

### IMU Fundamentals in Simulation

An Inertial Measurement Unit (IMU) typically contains:

- **Accelerometer**: Measures linear acceleration
- **Gyroscope**: Measures angular velocity
- **Magnetometer**: Measures magnetic field (for heading)

IMU simulation must model various error sources:

- **Bias**: Constant offset in measurements
- **Scale factor errors**: Proportional errors
- **Noise**: Random measurement variations
- **Drift**: Time-dependent bias changes
- **Temperature effects**: Performance variation with temperature

### IMU Simulation Implementation

```csharp
using UnityEngine;
using System;

public class IMUSimulator : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float accelerometerNoiseDensity = 0.002f;   // m/s^2/sqrt(Hz)
    public float gyroscopeNoiseDensity = 0.0001f;      // rad/s/sqrt(Hz)
    public float accelerometerRandomWalk = 0.001f;     // m/s^3/sqrt(Hz)
    public float gyroscopeRandomWalk = 0.00001f;       // rad/s^2/sqrt(Hz)
    public float magnetometerNoise = 0.1f;             // uT

    [Header("Bias Parameters")]
    public float accelerometerBiasStability = 0.01f;   // m/s^2
    public float gyroscopeBiasStability = 0.001f;      // rad/s
    public float biasCorrelationTime = 3600.0f;        // seconds

    [Header("Temperature Effects")]
    public float temperatureCoefficientAccel = 0.0001f; // (m/s^2)/degC
    public float temperatureCoefficientGyro = 0.00001f; // (rad/s)/degC
    public float ambientTemperature = 25.0f;           // degC

    private Vector3 trueAcceleration;
    private Vector3 trueAngularVelocity;
    private Vector3 trueMagneticField;
    private Vector3 accelerometerBias;
    private Vector3 gyroscopeBias;
    private Vector3 accelerometerWalk;
    private Vector3 gyroscopeWalk;
    private float lastUpdateTime;

    void Start()
    {
        InitializeIMU();
    }

    void InitializeIMU()
    {
        // Initialize biases with random values within stability limits
        accelerometerBias = new Vector3(
            UnityEngine.Random.Range(-accelerometerBiasStability, accelerometerBiasStability),
            UnityEngine.Random.Range(-accelerometerBiasStability, accelerometerBiasStability),
            UnityEngine.Random.Range(-accelerometerBiasStability, accelerometerBiasStability)
        );

        gyroscopeBias = new Vector3(
            UnityEngine.Random.Range(-gyroscopeBiasStability, gyroscopeBiasStability),
            UnityEngine.Random.Range(-gyroscopeBiasStability, gyroscopeBiasStability),
            UnityEngine.Random.Range(-gyroscopeBiasStability, gyroscopeBiasStability)
        );

        accelerometerWalk = Vector3.zero;
        gyroscopeWalk = Vector3.zero;
        lastUpdateTime = Time.time;
    }

    void Update()
    {
        UpdateIMUBias();
    }

    void UpdateIMUBias()
    {
        float deltaTime = Time.time - lastUpdateTime;
        lastUpdateTime = Time.time;

        // Update random walk components (first-order Gauss-Markov process)
        float decayFactor = Mathf.Exp(-deltaTime / biasCorrelationTime);

        accelerometerWalk += GetWhiteNoiseVector(accelerometerRandomWalk) * Mathf.Sqrt(deltaTime);
        gyroscopeWalk += GetWhiteNoiseVector(gyroscopeRandomWalk) * Mathf.Sqrt(deltaTime);

        // Apply decay to bias (bias instability over time)
        accelerometerBias *= decayFactor;
        gyroscopeBias *= decayFactor;

        // Add new random walk components
        accelerometerBias += accelerometerWalk * deltaTime;
        gyroscopeBias += gyroscopeWalk * deltaTime;
    }

    public sensor_msgs.Imu GetIMUData()
    {
        // Get true values from the simulation
        trueAcceleration = GetTrueAcceleration();
        trueAngularVelocity = GetTrueAngularVelocity();
        trueMagneticField = GetTrueMagneticField();

        // Apply sensor model
        Vector3 measuredAccel = ApplyAccelerometerModel(trueAcceleration);
        Vector3 measuredGyro = ApplyGyroscopeModel(trueAngularVelocity);
        Vector3 measuredMag = ApplyMagnetometerModel(trueMagneticField);

        // Create ROS IMU message
        sensor_msgs.Imu imuMsg = new sensor_msgs.Imu();

        // Fill acceleration data
        imuMsg.linear_acceleration = new geometry_msgs.Vector3(
            measuredAccel.x, measuredAccel.y, measuredAccel.z
        );

        // Fill angular velocity data
        imuMsg.angular_velocity = new geometry_msgs.Vector3(
            measuredGyro.x, measuredGyro.y, measuredGyro.z
        );

        // Estimate orientation from gravity and magnetic field
        imuMsg.orientation = EstimateOrientation(measuredAccel, measuredMag);

        // Set header information
        imuMsg.header.stamp = GetROSTime();
        imuMsg.header.frame_id = "imu_link";

        return imuMsg;
    }

    Vector3 ApplyAccelerometerModel(Vector3 trueAccel)
    {
        // Add temperature effect
        float temperatureEffect = (ambientTemperature - 25.0f) * temperatureCoefficientAccel;
        Vector3 tempEffectedAccel = trueAccel + Vector3.one * temperatureEffect;

        // Add bias
        Vector3 biasedAccel = tempEffectedAccel + accelerometerBias;

        // Add noise
        Vector3 noise = GetWhiteNoiseVector(accelerometerNoiseDensity);
        Vector3 noisyAccel = biasedAccel + noise;

        return noisyAccel;
    }

    Vector3 ApplyGyroscopeModel(Vector3 trueGyro)
    {
        // Add temperature effect
        float temperatureEffect = (ambientTemperature - 25.0f) * temperatureCoefficientGyro;
        Vector3 tempEffectedGyro = trueGyro + Vector3.one * temperatureEffect;

        // Add bias
        Vector3 biasedGyro = tempEffectedGyro + gyroscopeBias;

        // Add noise
        Vector3 noise = GetWhiteNoiseVector(gyroscopeNoiseDensity);
        Vector3 noisyGyro = biasedGyro + noise;

        return noisyGyro;
    }

    Vector3 ApplyMagnetometerModel(Vector3 trueMag)
    {
        // Add noise to magnetic field measurement
        Vector3 noise = GetWhiteNoiseVector(magnetometerNoise);
        return trueMag + noise;
    }

    Vector3 GetTrueAcceleration()
    {
        // Get true acceleration from physics simulation
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // True acceleration is the derivative of velocity plus gravity
            Vector3 gravity = Physics.gravity;
            return rb.velocity / Time.fixedDeltaTime + gravity;
        }

        // Fallback: assume zero acceleration if no rigidbody
        return Vector3.zero;
    }

    Vector3 GetTrueAngularVelocity()
    {
        // Get true angular velocity from physics simulation
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            return rb.angularVelocity;
        }

        // Fallback: assume zero angular velocity
        return Vector3.zero;
    }

    Vector3 GetTrueMagneticField()
    {
        // Return Earth's magnetic field (simplified)
        // In real implementation, this would account for local magnetic anomalies
        return new Vector3(0.22f, 0.0f, 0.45f); // ~45 degree inclination, 0.5 uT magnitude
    }

    Vector3 GetWhiteNoiseVector(float noiseDensity)
    {
        return new Vector3(
            RandomGaussian() * noiseDensity,
            RandomGaussian() * noiseDensity,
            RandomGaussian() * noiseDensity
        );
    }

    geometry_msgs.Quaternion EstimateOrientation(Vector3 accel, Vector3 mag)
    {
        // Simple orientation estimation from accelerometer and magnetometer
        // This is a simplified version - real implementation would use more sophisticated filtering

        // Normalize vectors
        Vector3 normalizedAccel = accel.normalized;
        Vector3 normalizedMag = mag.normalized;

        // Create coordinate system
        Vector3 zAxis = -normalizedAccel; // Accelerometer points opposite to gravity
        Vector3 xAxis = Vector3.Cross(normalizedMag, zAxis).normalized;
        Vector3 yAxis = Vector3.Cross(zAxis, xAxis);

        // Create rotation matrix and convert to quaternion
        Matrix4x4 rotationMatrix = new Matrix4x4();
        rotationMatrix.SetColumn(0, new Vector4(xAxis.x, yAxis.x, zAxis.x, 0));
        rotationMatrix.SetColumn(1, new Vector4(xAxis.y, yAxis.y, zAxis.y, 0));
        rotationMatrix.SetColumn(2, new Vector4(xAxis.z, yAxis.z, zAxis.z, 0));
        rotationMatrix.SetColumn(3, new Vector4(0, 0, 0, 1));

        Quaternion orientation = Quaternion.LookRotation(zAxis, yAxis);

        return new geometry_msgs.Quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        );
    }

    builtin_interfaces.Time GetROSTime()
    {
        // Return current ROS time
        double rosTime = Time.time;
        builtin_interfaces.Time time = new builtin_interfaces.Time();
        time.sec = (int)rosTime;
        time.nanosec = (uint)((rosTime - time.sec) * 1e9);
        return time;
    }

    float RandomGaussian()
    {
        float u1 = UnityEngine.Random.value;
        float u2 = UnityEngine.Random.value;

        if (u1 < Mathf.Epsilon) u1 = Mathf.Epsilon;

        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
    }
}
```

## Sensor Synchronization and Data Accuracy

### Time Synchronization Challenges

In multi-sensor systems, synchronization is critical for accurate perception. Key challenges include:

- **Clock drift**: Different sensors may have slightly different time bases
- **Latency variations**: Processing delays can vary between sensors
- **Update rates**: Different sensors may operate at different frequencies

### Synchronization Implementation

```csharp
using System.Collections.Generic;

public class SensorSynchronizer : MonoBehaviour
{
    [Header("Synchronization Parameters")]
    public float maxTimeDifference = 0.01f; // 10ms tolerance
    public int maxBufferSize = 100;         // Maximum messages to buffer

    private Dictionary<string, Queue<SensorMessage>> sensorBuffers;
    private List<SynchronizedData> synchronizedData;
    private float lastSynchronizationTime;

    [System.Serializable]
    public class SensorMessage
    {
        public string sensorType;
        public double timestamp;
        public object data;
    }

    [System.Serializable]
    public class SynchronizedData
    {
        public double timestamp;
        public Dictionary<string, object> sensorData;
    }

    void Start()
    {
        InitializeSynchronization();
    }

    void InitializeSynchronization()
    {
        sensorBuffers = new Dictionary<string, Queue<SensorMessage>>();
        synchronizedData = new List<SynchronizedData>();
        lastSynchronizationTime = Time.time;
    }

    public void AddSensorData(string sensorType, double timestamp, object data)
    {
        if (!sensorBuffers.ContainsKey(sensorType))
        {
            sensorBuffers[sensorType] = new Queue<SensorMessage>();
        }

        SensorMessage message = new SensorMessage
        {
            sensorType = sensorType,
            timestamp = timestamp,
            data = data
        };

        sensorBuffers[sensorType].Enqueue(message);

        // Limit buffer size
        if (sensorBuffers[sensorType].Count > maxBufferSize)
        {
            sensorBuffers[sensorType].Dequeue();
        }

        // Attempt synchronization
        AttemptSynchronization();
    }

    void AttemptSynchronization()
    {
        // Check if we have data from all sensors
        if (AllSensorsHaveData())
        {
            // Find the closest timestamps across all sensors
            double referenceTime = FindReferenceTimestamp();

            // Check if timestamps are within tolerance
            if (AreTimestampsWithinTolerance(referenceTime))
            {
                // Create synchronized data package
                SynchronizedData syncData = CreateSynchronizedData(referenceTime);

                if (syncData != null)
                {
                    synchronizedData.Add(syncData);

                    // Remove synchronized messages from buffers
                    RemoveSynchronizedMessages(referenceTime);
                }
            }
        }
    }

    bool AllSensorsHaveData()
    {
        // Check if all registered sensors have at least one message
        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count == 0)
                return false;
        }
        return sensorBuffers.Count > 0;
    }

    double FindReferenceTimestamp()
    {
        // Find the median timestamp across all sensors
        List<double> allTimestamps = new List<double>();

        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count > 0)
            {
                allTimestamps.Add(buffer.Peek().timestamp);
            }
        }

        allTimestamps.Sort();

        if (allTimestamps.Count == 0)
            return 0;

        int middleIndex = allTimestamps.Count / 2;
        return allTimestamps[middleIndex];
    }

    bool AreTimestampsWithinTolerance(double referenceTime)
    {
        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count > 0)
            {
                double timeDiff = Mathf.Abs((float)(buffer.Peek().timestamp - referenceTime));
                if (timeDiff > maxTimeDifference)
                    return false;
            }
        }
        return true;
    }

    SynchronizedData CreateSynchronizedData(double referenceTime)
    {
        SynchronizedData syncData = new SynchronizedData();
        syncData.timestamp = referenceTime;
        syncData.sensorData = new Dictionary<string, object>();

        foreach (var kvp in sensorBuffers)
        {
            string sensorType = kvp.Key;
            Queue<SensorMessage> buffer = kvp.Value;

            if (buffer.Count > 0)
            {
                SensorMessage message = buffer.Peek();
                syncData.sensorData[sensorType] = message.data;
            }
        }

        return syncData;
    }

    void RemoveSynchronizedMessages(double referenceTime)
    {
        foreach (var kvp in sensorBuffers)
        {
            Queue<SensorMessage> buffer = kvp.Value;

            if (buffer.Count > 0)
            {
                SensorMessage message = buffer.Peek();
                double timeDiff = Mathf.Abs((float)(message.timestamp - referenceTime));

                if (timeDiff <= maxTimeDifference)
                {
                    buffer.Dequeue();
                }
            }
        }
    }

    public List<SynchronizedData> GetSynchronizedData()
    {
        return synchronizedData;
    }

    public void ClearSynchronizedData()
    {
        synchronizedData.Clear();
    }
}
```

## Evaluating Sensor Realism and Limitations

### Sensor Fidelity Metrics

To evaluate the realism of sensor simulation, consider these metrics:

#### Accuracy Metrics

- **Absolute error**: Difference between simulated and real measurements
- **Precision**: Consistency of repeated measurements
- **Bias**: Systematic offset in measurements
- **Drift**: Time-dependent changes in bias

#### Performance Metrics

- **Update rate**: Frequency of sensor data generation
- **Latency**: Delay between physical event and sensor reading
- **Throughput**: Data processing capacity

### Validation Against Real Hardware

```csharp
public class SensorValidator : MonoBehaviour
{
    [Header("Validation Parameters")]
    public TextAsset realSensorData;  // CSV file with real sensor data
    public float validationThreshold = 0.1f;  // Acceptable error threshold

    private List<SensorReading> realData;
    private List<SensorReading> simulatedData;

    [System.Serializable]
    public class SensorReading
    {
        public double timestamp;
        public Vector3 data;  // For 3-axis sensors like IMU
        public float confidence;
    }

    void Start()
    {
        LoadRealSensorData();
    }

    void LoadRealSensorData()
    {
        if (realSensorData != null)
        {
            string[] lines = realSensorData.text.Split('\n');
            realData = new List<SensorReading>();

            foreach (string line in lines)
            {
                string[] values = line.Split(',');
                if (values.Length >= 4)  // timestamp, x, y, z
                {
                    SensorReading reading = new SensorReading
                    {
                        timestamp = double.Parse(values[0]),
                        data = new Vector3(
                            float.Parse(values[1]),
                            float.Parse(values[2]),
                            float.Parse(values[3])
                        )
                    };
                    realData.Add(reading);
                }
            }
        }
    }

    public float CalculateFidelityMetric()
    {
        if (realData == null || simulatedData == null ||
            realData.Count != simulatedData.Count)
        {
            return 0.0f;  // Cannot calculate with mismatched data
        }

        float totalError = 0.0f;
        int validComparisons = 0;

        for (int i = 0; i < Mathf.Min(realData.Count, simulatedData.Count); i++)
        {
            float error = Vector3.Distance(realData[i].data, simulatedData[i].data);
            totalError += error;
            validComparisons++;
        }

        if (validComparisons > 0)
        {
            float averageError = totalError / validComparisons;
            // Convert to fidelity score (0-1, where 1 is perfect)
            return Mathf.Clamp01(1.0f - (averageError / validationThreshold));
        }

        return 0.0f;
    }

    public void CompareWithRealData()
    {
        float fidelity = CalculateFidelityMetric();
        Debug.Log($"Sensor fidelity: {fidelity * 100:F2}%");

        if (fidelity < 0.8f)
        {
            Debug.LogWarning("Sensor simulation fidelity is below 80%. Consider adjusting noise parameters.");
        }
    }
}
```

## Best Practices for Sensor Simulation

### Modeling Guidelines

1. **Characterize Real Sensors**: Measure actual sensor noise, bias, and drift parameters
2. **Validate Against Hardware**: Compare simulation output with real sensor data
3. **Consider Environmental Factors**: Include temperature, humidity, and lighting effects
4. **Account for Sensor Placement**: Model mounting position and orientation errors

### Performance Considerations

- **Efficient Raycasting**: Use optimized algorithms for LiDAR simulation
- **Texture Compression**: Balance image quality with performance for RGB-D sensors
- **Update Rate Management**: Match simulation update rates to real sensor frequencies
- **Memory Management**: Efficiently handle large point cloud and image data

### Integration Strategies

- **Modular Design**: Create separate components for each sensor type
- **Calibration Support**: Include parameters for sensor calibration
- **ROS Message Compatibility**: Ensure proper message format for robotics frameworks
- **Data Logging**: Include facilities for recording and analyzing sensor data

## Summary

This chapter covered the implementation of realistic sensor simulation for digital twins in humanoid robotics. We explored LiDAR simulation with proper noise modeling, depth camera and RGB-D pipeline implementation, IMU modeling with drift characteristics, and sensor synchronization techniques. The fidelity of sensor simulation is crucial for the transferability of algorithms from simulation to reality, and proper validation against real hardware ensures that the digital twin accurately represents the physical system.
