---
sidebar_position: 1
title: 'Chapter 1: Voice-to-Action Interfaces'
---

Chapter 1: Voice-to-Action Interfaces

## Learning Objectives

By the end of this chapter, you will be able to:

- Integrate OpenAI Whisper for speech-to-text conversion
- Convert voice commands into structured task intents
- Publish intents to ROS 2 topics/actions
- Create voice command processing pipelines
- Implement troubleshooting strategies for Whisper integration

## Key Topics

### 1. OpenAI Whisper Integration

OpenAI Whisper is a state-of-the-art speech recognition model that can convert spoken language into text. For robotics applications, Whisper provides a robust foundation for voice command processing.

#### Installation and Setup

To get started with Whisper in your robotics project:

```bash
pip install openai-whisper
# Or for GPU acceleration
pip install openai-whisper[cuda]
```

#### Basic Whisper Implementation

```python
import whisper
import torch
import rospy
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData

class WhisperVoiceProcessor:
    def __init__(self, model_size="base"):
        # Load the Whisper model
        self.model = whisper.load_model(model_size)

        # Initialize ROS
        rospy.init_node('whisper_voice_processor')

        # Subscribe to audio data
        self.audio_sub = rospy.Subscriber('/audio_input', AudioData, self.audio_callback)

        # Publisher for recognized text
        self.text_pub = rospy.Publisher('/voice_commands', String, queue_size=10)

        rospy.loginfo("Whisper voice processor initialized")

    def audio_callback(self, audio_data):
        """Process incoming audio data"""
        # Convert audio data to numpy array
        audio_array = self.audio_to_numpy(audio_data)

        # Transcribe the audio
        result = self.model.transcribe(audio_array)
        recognized_text = result['text']

        # Publish the recognized text
        self.text_pub.publish(recognized_text)
        rospy.loginfo(f"Recognized: {recognized_text}")

    def audio_to_numpy(self, audio_data):
        """Convert ROS AudioData to numpy array"""
        # Implementation depends on audio format
        # This is a simplified example
        import numpy as np
        audio_array = np.frombuffer(audio_data.data, dtype=np.int16)
        return audio_array.astype(np.float32) / 32768.0

if __name__ == '__main__':
    processor = WhisperVoiceProcessor()
    rospy.spin()
```

### 2. Speech-to-Text Conversion Processes

The speech-to-text conversion process involves several steps to ensure accurate recognition:

1. **Audio Preprocessing**: Clean and normalize audio input
2. **Feature Extraction**: Extract relevant features from the audio signal
3. **Model Inference**: Use the Whisper model to transcribe speech
4. **Post-processing**: Clean and format the transcribed text

#### Audio Preprocessing Example

```python
import numpy as np
from scipy import signal

def preprocess_audio(audio_data, sample_rate=16000):
    """Preprocess audio data for better recognition"""
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Apply pre-emphasis filter
    audio_data = signal.lfilter([1, -0.97], [1], audio_data)

    # Trim silence at the beginning and end
    threshold = 0.01  # 1% of max amplitude
    non_silent = np.where(np.abs(audio_data) > threshold)[0]

    if len(non_silent) > 0:
        start = non_silent[0]
        end = non_silent[-1] + 1
        audio_data = audio_data[start:end]

    return audio_data
```

### 3. Voice Command Processing Pipelines

Creating an effective voice command processing pipeline involves multiple stages:

#### Pipeline Architecture

```python
class VoiceCommandPipeline:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.intent_classifier = IntentClassifier()
        self.ros_publisher = ROSPublisher()

    def process_voice_command(self, audio_data):
        """Complete pipeline for processing voice commands"""
        # Step 1: Convert audio to text
        text = self.audio_to_text(audio_data)

        # Step 2: Classify intent
        intent = self.intent_classifier.classify(text)

        # Step 3: Execute ROS action
        self.ros_publisher.publish_intent(intent)

        return intent

    def audio_to_text(self, audio_data):
        """Convert audio to text using Whisper"""
        # Preprocess audio
        processed_audio = preprocess_audio(audio_data)

        # Transcribe using Whisper
        result = self.whisper_model.transcribe(processed_audio)
        return result['text']
```

### 4. Structured Task Intents Mapping

To convert voice commands into actionable tasks, we need to map recognized text to structured intents:

#### Intent Mapping Example

```python
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TaskIntent:
    action: str
    parameters: Dict[str, any]
    confidence: float

class IntentClassifier:
    def __init__(self):
        self.intent_patterns = {
            'move_to_location': [
                r'move to (.+)',
                r'go to (.+)',
                r'go to the (.+)',
                r'walk to (.+)'
            ],
            'pick_object': [
                r'pick up (.+)',
                r'grab (.+)',
                r'get the (.+)',
                r'take (.+)'
            ],
            'place_object': [
                r'place (.+) at (.+)',
                r'put (.+) on (.+)',
                r'drop (.+) at (.+)'
            ],
            'stop_robot': [
                r'stop',
                r'freeze',
                r'hold position'
            ]
        }

    def classify(self, text: str) -> TaskIntent:
        """Classify text into structured intent"""
        text_lower = text.lower().strip()

        for action, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    # Extract parameters based on pattern groups
                    params = {
                        f'param_{i+1}': param
                        for i, param in enumerate(match.groups())
                    }
                    return TaskIntent(action=action, parameters=params, confidence=0.9)

        # If no pattern matches, return a generic command
        return TaskIntent(action='unknown', parameters={'text': text}, confidence=0.0)
```

### 5. Practical Implementation: Voice-to-Action System

Here's a complete implementation of a voice-to-action system:

```python
#!/usr/bin/env python3
import rospy
import whisper
import numpy as np
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal
import json

class VoiceToActionSystem:
    def __init__(self):
        rospy.init_node('voice_to_action_system')

        # Load Whisper model
        self.model = whisper.load_model("base")

        # Subscribe to voice commands
        self.voice_sub = rospy.Subscriber('/voice_input', String, self.voice_callback)

        # Publishers for different action types
        self.nav_pub = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10)
        self.command_pub = rospy.Publisher('/robot_commands', String, queue_size=10)

        # Location mapping for navigation commands
        self.location_map = {
            'kitchen': [1.0, 2.0, 0.0],  # x, y, theta
            'living room': [3.0, 1.0, 1.57],
            'bedroom': [0.0, 4.0, 3.14],
            'office': [2.5, 3.5, -1.57]
        }

        rospy.loginfo("Voice-to-Action system initialized")

    def voice_callback(self, msg):
        """Process incoming voice command"""
        command = msg.data.lower()

        # Process the command and execute appropriate action
        if self.is_navigation_command(command):
            self.handle_navigation_command(command)
        elif self.is_action_command(command):
            self.handle_action_command(command)
        else:
            rospy.logwarn(f"Unknown command: {command}")

    def is_navigation_command(self, command):
        """Check if command is a navigation command"""
        navigation_keywords = ['go to', 'move to', 'walk to', 'navigate to']
        return any(keyword in command for keyword in navigation_keywords)

    def is_action_command(self, command):
        """Check if command is an action command"""
        action_keywords = ['pick', 'grab', 'take', 'place', 'put', 'drop']
        return any(keyword in command for keyword in action_keywords)

    def handle_navigation_command(self, command):
        """Handle navigation commands"""
        # Extract location from command
        for location_name, location_coords in self.location_map.items():
            if location_name in command:
                self.navigate_to_location(location_name, location_coords)
                return

        rospy.logwarn(f"Unknown location in command: {command}")

    def navigate_to_location(self, location_name, coords):
        """Navigate to a specific location"""
        goal = MoveBaseActionGoal()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"

        goal.goal.target_pose.header.frame_id = "map"
        goal.goal.target_pose.pose.position.x = coords[0]
        goal.goal.target_pose.pose.position.y = coords[1]
        goal.goal.target_pose.pose.orientation.z = np.sin(coords[2] / 2)
        goal.goal.target_pose.pose.orientation.w = np.cos(coords[2] / 2)

        self.nav_pub.publish(goal)
        rospy.loginfo(f"Navigating to {location_name}")

    def handle_action_command(self, command):
        """Handle action commands"""
        action_msg = String()
        action_msg.data = command
        self.command_pub.publish(action_msg)
        rospy.loginfo(f"Action command published: {command}")

    def run(self):
        """Run the voice-to-action system"""
        rospy.spin()

if __name__ == '__main__':
    system = VoiceToActionSystem()
    system.run()
```

### 6. Troubleshooting Guide for Whisper Integration

Common issues and solutions when integrating Whisper:

#### Audio Format Issues

- **Problem**: Whisper expects audio in specific formats
- **Solution**: Ensure audio is at the correct sample rate (16kHz recommended)

#### Performance Issues

- **Problem**: Slow transcription on CPU
- **Solution**: Use GPU acceleration or smaller Whisper models

#### Accuracy Issues

- **Problem**: Poor recognition in noisy environments
- **Solution**: Implement audio preprocessing and noise reduction

#### ROS Integration Issues

- **Problem**: Latency in voice processing pipeline
- **Solution**: Optimize audio buffering and implement asynchronous processing

## Assessment Criteria

- Students can successfully integrate OpenAI Whisper for speech-to-text conversion
- Students can convert voice commands into structured task intents
- Students can publish intents to ROS 2 topics/actions
- Students can create effective voice command processing pipelines
- Students can troubleshoot common issues in Whisper integration
