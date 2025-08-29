# Perception Module for Skillgraph G1 Task

## Overview

This repository contains the perception system for the Skillgraph G1 task, designed to detect and track objects using RealSense cameras for robotic manipulation tasks.

## Authors and Credits

- **Original Code**: CMU RI ICL (ICRA WBCD competition)
- **Task-Specific Adaptation**: Han Zhou (ICL)
- **ROS Interface**: Yijie Liao (Summer Research Intern, ICL)
- **README Author**: Yijie Liao

For specific usage questions, please contact Yijie Liao or Han Zhou.

## Setup

### Prerequisites
- Conda package manager
- Python environment

### Installation
```bash
# Create and activate a conda environment
# Replace <env_name> with any name you prefer
conda create -n <env_name> python=3.11 -y   # Python 3.11+ recommended
conda activate <env_name>

# Install dependencies as needed when running the code
# You can install packages on-demand if you encounter missing imports

```

## Module Description

**Note**: These files are also used in the G1-grasp repository.

### WBCD Task Modules

#### 1. `bread_wbcd.py`
**Purpose**: Initial object detection and localization

**Functionality**:
- Detects object locations in format: `[[x,y,z], yaw, object_name]`
- Identifies positions of box, box_pack, and objects visible to RealSense camera
- **Active Mode**: `Demo == "all"`

**Usage Pattern**:
```python
# Pseudo-code for code logic
for i in range(10):
    capture_frame()
    process_detections()
    #send the 10th frame
    if i == 9:
        return stable_object_positions   
        # use 10th frame as initialization
```

#### 2. `bread_fail_recovery.py`
**Purpose**: Real-time object tracking for failure recovery

**Functionality**:
- Similar to `bread_wbcd.py` but provides continuous real-time updates
- Used for failure recovery and object relocation during task execution
- **Active Mode**: `Demo == "all"`

**Key Difference**: Sends real-time object location data instead of only the 10th frame

### Water Pouring Task Modules

#### 3. `bread_cup.py`
**Purpose**: Cup detection for pouring tasks

**Functionality**:
- Similar to `bread_wbcd.py` but specialized for cup detection
- Identifies different types of cups for water pouring scenarios
- **Active Mode**: `Demo == "cup"`

#### 4. `bread_cup_recovery.py`
**Purpose**: Real-time tracking for pouring task recovery

**Functionality**:
- Real-time location tracking including "lollipop" objects
- Simulates water pouring detection for cleanup operations
- **Active Mode**: `Demo == "cup"`

**Special Feature**: Tracks "lollipop" objects to simulate water pouring behavior

## Usage Examples

### Basic Object Detection
```python
# For initial object detection
python bread_wbcd.py  # Set Demo = "all" in code

# For cup detection
python bread_cup.py   # Set Demo = "cup" in code
```

### Real-time Tracking
```python
# For failure recovery
python bread_fail_recovery.py     # Set Demo = "all" in code

# For pouring task recovery
python bread_cup_recovery.py      # Set Demo = "cup" in code
```

## Configuration

Each module operates based on the `Demo` variable setting:
- `Demo == "all"`: Activates WBCD task modules
- `Demo == "cup"`: Activates water pouring task modules

## Output Format

All modules return object information in the format:
```python
[[x, y, z], yaw, object_name]
```

Where:
- `[x, y, z]`: 3D coordinates of the object
- `yaw`: Rotation angle
- `object_name`: String identifier for the detected object

## Integration

This perception system integrates with:
- G1-grasp repository
- ROS navigation and manipulation systems
- WBCD competition framework

## Notes

- The system waits for stable readings (10 frames) before returning initial positions
- Real-time modules provide continuous updates for dynamic task execution
- All detection is performed using RealSense camera data