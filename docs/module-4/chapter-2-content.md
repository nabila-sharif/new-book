---
sidebar_position: 2
title: 'Chapter 2: Cognitive Planning with LLMs'
---

Chapter 2: Cognitive Planning with LLMs

## Learning Objectives

By the end of this chapter, you will be able to:

- Integrate LLMs for cognitive planning in robotics applications
- Apply goal decomposition and task sequencing techniques
- Translate natural language goals into ROS 2 action graphs
- Create effective prompt engineering examples for robotics
- Implement error handling and replanning strategies
- Include safety constraints in cognitive planning systems

## Key Topics

### 1. LLM Integration for Cognitive Planning

Large Language Models (LLMs) can serve as cognitive planners that interpret high-level goals and generate executable action sequences for robots. This approach enables natural language interaction with robotic systems.

#### Basic LLM Integration

```python
import openai
import json
from typing import List, Dict, Any
import rospy

class LLMBasedPlanner:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.ros_actions = self.get_available_ros_actions()

    def get_available_ros_actions(self) -> List[Dict[str, Any]]:
        """Define available ROS actions for the LLM to use"""
        return [
            {
                "name": "move_to",
                "description": "Move robot to a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number", "description": "X coordinate"},
                        "y": {"type": "number", "description": "Y coordinate"},
                        "theta": {"type": "number", "description": "Orientation in radians"}
                    },
                    "required": ["x", "y", "theta"]
                }
            },
            {
                "name": "pick_object",
                "description": "Pick up an object at the current location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to pick"}
                    },
                    "required": ["object_name"]
                }
            },
            {
                "name": "place_object",
                "description": "Place the currently held object at the current location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {"type": "string", "description": "Name of the object to place"}
                    },
                    "required": ["object_name"]
                }
            }
        ]

    def plan_from_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Generate a plan from a natural language goal"""
        prompt = f"""
        You are a cognitive planner for a humanoid robot. Given the following goal,
        break it down into a sequence of executable actions.
        The available actions are: {json.dumps(self.ros_actions)}

        Goal: {goal}

        Return the plan as a JSON list of actions, where each action has:
        - name: the action name
        - parameters: the action parameters

        Example response:
        [
            {{"name": "move_to", "parameters": {{"x": 1.0, "y": 2.0, "theta": 0.0}}},
            {{"name": "pick_object", "parameters": {{"object_name": "red_cup"}}}
        ]
        """

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )

        plan_text = response.choices[0].message.content
        try:
            # Extract JSON from the response
            start_idx = plan_text.find('[')
            end_idx = plan_text.rfind(']') + 1
            plan_json = plan_text[start_idx:end_idx]
            plan = json.loads(plan_json)
            return plan
        except json.JSONDecodeError:
            rospy.logerr(f"Failed to parse plan: {plan_text}")
            return []
```

### 2. Goal Decomposition Techniques

Goal decomposition is crucial for breaking down complex tasks into manageable subtasks that the robot can execute.

#### Hierarchical Task Decomposition

```python
class GoalDecomposer:
    def __init__(self):
        self.subgoal_templates = {
            "complex_navigation": [
                "find_path_to_destination",
                "navigate_safely",
                "reach_destination"
            ],
            "object_manipulation": [
                "locate_object",
                "approach_object",
                "grasp_object",
                "transport_object",
                "place_object"
            ],
            "multi_room_task": [
                "navigate_to_room",
                "perform_task_in_room",
                "return_to_base"
            ]
        }

    def decompose_goal(self, goal: str) -> List[str]:
        """Decompose a high-level goal into subgoals"""
        # This is a simplified example - in practice, you'd use an LLM for this
        if "kitchen" in goal.lower() and ("get" in goal.lower() or "bring" in goal.lower()):
            return [
                "navigate to kitchen",
                "locate requested item",
                "grasp the item",
                "return to original location",
                "deliver the item"
            ]

        elif "clean" in goal.lower():
            return [
                "identify dirty areas",
                "navigate to first area",
                "clean the area",
                "check if area is clean",
                "move to next area if needed"
            ]

        else:
            # For unknown goals, return the original goal as a single subgoal
            return [goal]
```

### 3. Task Sequencing with LLMs

Sequencing tasks properly is essential for successful robot execution:

#### Sequential Task Planner

```python
class SequentialTaskPlanner:
    def __init__(self):
        self.dependencies = {
            "grasp_object": ["approach_object"],
            "place_object": ["transport_object"],
            "transport_object": ["grasp_object"],
            "navigate_safely": ["find_path_to_destination"]
        }

    def order_tasks(self, tasks: List[str]) -> List[str]:
        """Order tasks based on dependencies"""
        ordered_tasks = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            for task in remaining_tasks:
                # Check if all dependencies are satisfied
                deps_satisfied = True
                if task in self.dependencies:
                    for dep in self.dependencies[task]:
                        if dep not in ordered_tasks:
                            deps_satisfied = False
                            break

                if deps_satisfied:
                    ordered_tasks.append(task)
                    remaining_tasks.remove(task)
                    break
            else:
                # If no task can be added, there might be circular dependencies
                rospy.logwarn(f"Could not satisfy dependencies for remaining tasks: {remaining_tasks}")
                break

        return ordered_tasks
```

### 4. Prompt Engineering Examples for Robotics

Effective prompt engineering is crucial for getting reliable responses from LLMs in robotics contexts:

#### Context-Aware Prompting

```python
class RobotPromptEngineer:
    def __init__(self):
        self.robot_capabilities = {
            "navigation": True,
            "manipulation": True,
            "perception": True,
            "speech": True
        }
        self.environment = {
            "rooms": ["kitchen", "living room", "bedroom", "office"],
            "objects": ["cup", "book", "phone", "keys", "bottle"],
            "locations": {
                "kitchen": {"x": 1.0, "y": 2.0},
                "living room": {"x": 3.0, "y": 1.0},
                "bedroom": {"x": 0.0, "y": 4.0},
                "office": {"x": 2.5, "y": 3.5}
            }
        }

    def create_contextual_prompt(self, goal: str) -> str:
        """Create a contextual prompt for the LLM"""
        return f"""
        You are a cognitive planner for a humanoid robot with the following capabilities:
        {json.dumps(self.robot_capabilities)}

        The environment contains these rooms:
        {json.dumps(self.environment['rooms'])}

        The environment contains these objects:
        {json.dumps(self.environment['objects'])}

        Room locations:
        {json.dumps(self.environment['locations'])}

        Given the goal: "{goal}"

        Generate a sequence of actions that the robot can execute to achieve this goal.
        Consider the environment layout and robot capabilities.
        Each action should be specific and executable.

        Format your response as a JSON list of actions with the following structure:
        [
            {{
                "action": "action_name",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "reasoning": "Brief explanation of why this action is needed"
            }}
        ]

        Actions available:
        - move_to: Move to a specific location (x, y coordinates)
        - locate_object: Find an object in the environment
        - grasp_object: Pick up an object
        - place_object: Place an object at current location
        - speak: Make the robot speak a message
        """
```

### 5. Error Handling and Replanning Strategies

Robots often encounter unexpected situations that require replanning:

#### Robust Planning with Error Handling

```python
class RobustPlanner:
    def __init__(self):
        self.max_replan_attempts = 3
        self.error_recovery_strategies = {
            "navigation_failure": ["use_alternative_path", "request_assistance"],
            "object_not_found": ["search_alternative_locations", "ask_for_help"],
            "grasp_failure": ["retry_grasp", "use_different_approach", "abandon_task"]
        }

    def execute_plan_with_error_handling(self, plan: List[Dict], robot_state: Dict):
        """Execute a plan with error handling and replanning"""
        current_step = 0

        while current_step < len(plan):
            action = plan[current_step]

            try:
                # Execute the action
                success = self.execute_action(action, robot_state)

                if success:
                    current_step += 1
                    rospy.loginfo(f"Completed action: {action['action']}")
                else:
                    # Handle failure
                    rospy.logwarn(f"Action failed: {action['action']}")
                    recovery_success = self.handle_action_failure(
                        action, robot_state, plan, current_step
                    )

                    if not recovery_success:
                        rospy.logerr("Could not recover from action failure")
                        return False

            except Exception as e:
                rospy.logerr(f"Exception during action execution: {e}")
                return False

        return True

    def handle_action_failure(self, failed_action: Dict, robot_state: Dict,
                             plan: List[Dict], step_index: int) -> bool:
        """Handle action failure with potential replanning"""
        error_type = self.classify_error(failed_action, robot_state)

        if error_type in self.error_recovery_strategies:
            strategies = self.error_recovery_strategies[error_type]

            for strategy in strategies:
                if self.apply_recovery_strategy(strategy, failed_action, robot_state, plan, step_index):
                    return True

        # If all recovery strategies fail, try replanning
        return self.replan_after_failure(failed_action, robot_state, plan, step_index)

    def replan_after_failure(self, failed_action: Dict, robot_state: Dict,
                            plan: List[Dict], step_index: int) -> bool:
        """Replan the remaining tasks after a failure"""
        rospy.loginfo("Attempting to replan after failure...")

        # For simplicity, this is a basic replanning
        # In practice, you'd use the LLM to generate a new plan
        remaining_goal = self.extract_remaining_goal(plan, step_index)

        if remaining_goal:
            new_plan = self.generate_new_plan(remaining_goal, robot_state)
            if new_plan:
                # Execute the new plan
                return self.execute_plan_with_error_handling(new_plan, robot_state)

        return False
```

### 6. Safety Constraints Implementation

Safety is paramount in robotics applications:

#### Safety-Constrained Planning

```python
class SafetyConstrainedPlanner:
    def __init__(self):
        self.safety_constraints = {
            "no_go_zones": [],  # List of coordinates to avoid
            "speed_limits": {"indoor": 0.5, "outdoor": 1.0},  # m/s
            "object_handling": {"fragile_objects": ["glass", "ceramic"]},
            "human_proximity": {"minimum_distance": 0.5}  # meters
        }

    def validate_plan_safety(self, plan: List[Dict]) -> bool:
        """Validate that a plan meets safety constraints"""
        for action in plan:
            if not self.is_action_safe(action):
                return False
        return True

    def is_action_safe(self, action: Dict) -> bool:
        """Check if a single action is safe"""
        action_type = action.get("action", "")

        if action_type == "move_to":
            target_location = action.get("parameters", {}).get("x", 0), action.get("parameters", {}).get("y", 0)
            return self.is_navigation_safe(target_location)

        elif action_type == "grasp_object":
            obj_name = action.get("parameters", {}).get("object_name", "")
            return self.is_object_safe_to_grasp(obj_name)

        # Add more safety checks for other action types
        return True

    def is_navigation_safe(self, location: tuple) -> bool:
        """Check if navigation to a location is safe"""
        # Check if location is in no-go zone
        for no_go_zone in self.safety_constraints["no_go_zones"]:
            if self.distance(location, no_go_zone) < 0.5:  # 0.5m threshold
                return False
        return True

    def is_object_safe_to_grasp(self, obj_name: str) -> bool:
        """Check if it's safe to grasp an object"""
        fragile_objects = self.safety_constraints["object_handling"]["fragile_objects"]
        return obj_name.lower() not in [obj.lower() for obj in fragile_objects]

    def distance(self, pos1: tuple, pos2: tuple) -> float:
        """Calculate Euclidean distance between two points"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
```

### 7. Complete Cognitive Planning System

Here's a complete implementation combining all the components:

```python
#!/usr/bin/env python3
import rospy
import openai
import json
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import time

class CompleteCognitivePlanner:
    def __init__(self, openai_api_key: str):
        rospy.init_node('cognitive_planner')
        openai.api_key = openai_api_key

        # Initialize components
        self.llm_planner = LLMBasedPlanner(openai_api_key)
        self.goal_decomposer = GoalDecomposer()
        self.task_sequencer = SequentialTaskPlanner()
        self.prompt_engineer = RobotPromptEngineer()
        self.robust_planner = RobustPlanner()
        self.safety_planner = SafetyConstrainedPlanner()

        # Publishers and subscribers
        self.goal_sub = rospy.Subscriber('/natural_language_goals', String, self.goal_callback)
        self.action_pub = rospy.Publisher('/robot_action_queue', String, queue_size=10)
        self.status_pub = rospy.Publisher('/planning_status', String, queue_size=10)

        rospy.loginfo("Cognitive Planning System initialized")

    def goal_callback(self, msg):
        """Process incoming natural language goals"""
        goal = msg.data
        rospy.loginfo(f"Received goal: {goal}")

        # Publish planning status
        status_msg = String()
        status_msg.data = f"Planning for goal: {goal}"
        self.status_pub.publish(status_msg)

        # Plan the task
        plan = self.plan_for_goal(goal)

        if plan:
            # Validate plan safety
            if self.safety_planner.validate_plan_safety(plan):
                # Execute the plan with error handling
                success = self.robust_planner.execute_plan_with_error_handling(plan, {})

                if success:
                    rospy.loginfo("Plan executed successfully")
                    status_msg.data = "Plan completed successfully"
                else:
                    rospy.logerr("Plan execution failed")
                    status_msg.data = "Plan execution failed"
            else:
                rospy.logerr("Plan failed safety validation")
                status_msg.data = "Plan failed safety validation"
        else:
            rospy.logerr("Could not generate plan for goal")
            status_msg.data = "Could not generate plan"

        self.status_pub.publish(status_msg)

    def plan_for_goal(self, goal: str) -> List[Dict]:
        """Generate a plan for a given goal"""
        # Decompose the goal
        subgoals = self.goal_decomposer.decompose_goal(goal)

        # Generate plan for each subgoal
        full_plan = []
        for subgoal in subgoals:
            # Use LLM to generate detailed plan for subgoal
            subgoal_plan = self.llm_planner.plan_from_goal(subgoal)
            full_plan.extend(subgoal_plan)

        # Order tasks based on dependencies
        ordered_plan = self.task_sequencer.order_tasks(full_plan)

        return ordered_plan

    def run(self):
        """Run the cognitive planning system"""
        rospy.spin()

if __name__ == '__main__':
    # You would need to provide your OpenAI API key here
    # planner = CompleteCognitivePlanner("your-api-key-here")
    # planner.run()
    pass
```

## Assessment Criteria

- Students can integrate LLMs for cognitive planning in robotics applications
- Students can apply effective goal decomposition and task sequencing techniques
- Students can translate natural language goals into executable ROS 2 action sequences
- Students can create well-engineered prompts for robotics applications
- Students can implement robust error handling and replanning strategies
- Students can incorporate safety constraints into cognitive planning systems
