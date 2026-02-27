# ROS 2 Integration Learning Guide

**Time to Complete:** 1-2 hours  
**What You'll Build:** ROS 2 publishers that broadcast face recognition results to the robot  
**Learning Method:** Implement code step-by-step by filling in TODOs

---

## Overview

### What You'll Learn

1. **Inter-Process Communication** - Why separate robot programs need a shared message bus
2. **ROS 2 Nodes and Topics** - Registering your program and creating named channels
3. **Publishing Messages** - Sending recognition results so other robot components can react
4. **Conditional Integration** - Making ROS optional so the code still works on your laptop

### The Big Picture

In Semesters 1 and 2, your pipeline ended at the screen:

```
Camera → Detection → Crop → ┬→ Identity → Name      ─┐
                             │                         ├→ cv2.putText (screen only)
                             └→ Emotion  → Label      ─┘
```

Now you're adding an **output layer** so the rest of the robot can hear the results:

```
Camera → Detection → Crop → ┬→ Identity → Name      ─┬→ cv2.putText (screen)
                             │                         │
                             └→ Emotion  → Label      ─┼→ ROS 2 Publisher
                                                       │
                             Face Position → (x,y)    ─┘
                                                        ↓
                                          ┌──────────────────────────┐
                                          │     ROS 2 Topics         │
                                          │                          │
                                          │  /face/identity  ──→ Behavior Node
                                          │  /face/emotion   ──→ Behavior Node
                                          │  /face/position  ──→ Navigation Node
                                          └──────────────────────────┘
```

**Key Insight:** Your recognition code stays identical. You're just adding a few lines that broadcast the results over ROS 2, like plugging a radio transmitter into an existing system.

### Why ROS?

Your robot runs multiple programs simultaneously:
- Face recognition (your code)
- Arm control
- Navigation / wheel control
- Maybe speech

These are **separate Python processes**. They can't share variables because they each have their own memory. ROS 2 is a **message bus** that lets them send data to each other.

Without ROS, your recognition results only exist inside your program. With ROS, any other program on the robot can subscribe and react.

---

## Core Concepts (Read This First!)

### Concept 9: Inter-Process Communication

Imagine two Python scripts running in separate terminals:

```
Terminal 1 (your code):          Terminal 2 (arm controller):
┌─────────────────────┐         ┌─────────────────────┐
│ name = "Ben"        │         │ # How do I get      │
│ emotion = "Happy"   │   ???   │ # the name from     │
│ position = (320,200)│ ──────→ │ # Terminal 1?       │
└─────────────────────┘         └─────────────────────┘
```

Terminal 2 **cannot** do `from face_recognizer import name`. That variable lives in a different process's memory. It's like two people in separate rooms -- one knows the answer, but the other can't hear them.

**Solutions:**
- Write to a file, other process reads it (slow, messy)
- Use sockets (works, but you write all the plumbing)
- Use ROS topics (standardized, every robot component speaks the same language)

ROS is a **message board** between programs:

```
┌─────────────────────┐
│    ROS 2 Message     │
│       Board          │
│                      │
│  /face/identity:     │
│    "ben|0.85"        │  ← Your code posts here
│                      │
│  /face/emotion:      │
│    "Happiness"       │  ← Your code posts here
│                      │
│  /face/position:     │
│    "320|200|80|100"  │  ← Your code posts here
│                      │
└──────────┬───────────┘
           │
    Anyone can read
           │
    ┌──────┴──────┐
    │ Arm control │  → reads /face/identity → waves at Ben
    │ Navigation  │  → reads /face/position → turns toward face
    └─────────────┘
```

### Concept 10: Nodes, Topics, and Publishers

Three terms to know:

**Node** = Your program, registered with ROS.
```python
rclpy.init()                              # Connect to ROS 2
ros_node = Node('face_recognition_node')  # Register with a name
```
When someone runs `ros2 node list`, they'll see `face_recognition_node`.

**Topic** = A named channel for messages.
```
/face/identity   ← channel for identity results
/face/emotion    ← channel for emotion results
/face/position   ← channel for face position
```
Topics are just strings. You pick the name. Convention uses `/` like file paths.

**Publisher** = The thing that posts messages to a topic.
```python
pub = node.create_publisher(String, '/face/identity', 10)
```
This says: "I want to post `String` messages to the `/face/identity` channel, buffer up to 10."

**Subscriber** = The thing that reads messages from a topic (other nodes do this, not you).
```python
# In someone else's code:
node.create_subscription(String, '/face/identity', callback, 10)
```

**The full flow:**
```
Your code:           ROS 2:              Other node:
                  ┌──────────┐
publish("ben") →  │  /face/  │  → callback("ben")
                  │ identity │
                  └──────────┘
```

### Concept 11: Message Types

Messages need a format so both sides agree on what's being sent.

**String** (from `std_msgs.msg`) is the simplest -- just text:
```python
from std_msgs.msg import String

msg = String()
msg.data = "ben|0.85"    # Any text you want
publisher.publish(msg)
```

**Why pipe-delimited strings?**
- Simple to create: `f'{name}|{similarity:.2f}'`
- Simple to parse: `name, sim = msg.data.split('|')`
- Good enough for our use case

**Contrast with Semester 1/2:**
- Semester 1: Results stayed as Python variables (screen only)
- Semester 2: Same -- variables displayed via `cv2.putText`
- Now: Results also leave the program as ROS messages

**Note:** ROS has many message types (Int32, Float64, Pose, Image, custom types). For our needs, String is sufficient. You can upgrade to custom messages later if the receiving node needs structured data.

---

## Setup (10 minutes)

### Step 1: Verify ROS 2 Humble Installation

On the Jetson (or wherever ROS 2 is installed):

```bash
source /opt/ros/humble/setup.bash
ros2 --version
# Should show: ros2 0.x.x (any version)
```

```bash
python3 -c "import rclpy; print('rclpy available')"
# Should print: rclpy available
```

If `rclpy` is not found, ROS 2 Humble is not installed or not sourced. Ask your team lead.

### Step 2: Understand What Changes

Only **one file** changes: `core/face_recognizer.py`

```
core/face_recognizer.py
│
├── Line ~24:  TODO 24 - Import ROS 2 libraries (3 new imports)
│
├── Line ~40:  TODO 25 - Add ros_node parameter to __init__
│   Line ~125:          - Create 3 publishers
│
├── Line ~873: TODO 26 - Publish results inside run_webcam3 loop
│
└── Line ~903: TODO 27 - Initialize ROS 2 in __main__ block
```

No other files change. Your detection, recognition, emotion, smoothing, and tracking code stays identical.

---

## Phase 1: ROS Setup (30-45 minutes)

### Concept: Optional Dependencies

A good practice for robot code is making ROS **optional**. This way:
- On your laptop (no ROS): code works normally, just no publishing
- On the Jetson (with ROS): code works AND publishes to the robot

The pattern is:
```python
def __init__(self, ..., ros_node=None):
    self.ros_node = ros_node
    if self.ros_node:
        # Create publishers (only if ROS is active)
        ...
```

If `ros_node` is `None`, the publishing code is skipped entirely.

### Implementation

**File:** `core/face_recognizer.py`

#### TODO 24: Import ROS 2 Libraries

Navigate to `core/face_recognizer.py` and find:

```python
import time

# TODO 24: Import ROS 2 libraries
```

**What to implement:**
1. Import `rclpy` (the ROS 2 Python client library)
2. Import `Node` from `rclpy.node` (to create a node object)
3. Import `String` from `std_msgs.msg` (the message type)

**Reasoning:**
- `rclpy` is the ROS 2 equivalent of `import rospy` in ROS 1
- `Node` is the class your program uses to register with ROS and create publishers
- `String` is the simplest message type -- just text data

**Hint:** These are standard ROS 2 imports. Every ROS 2 Python node starts with them.

#### TODO 25: Create ROS 2 Publishers

This TODO has two parts in the same file.

**Part A:** Find the `__init__` method signature:

```python
def __init__(self, 
             reference_path='models/reference_embeddings.npy',
             labels_path='models/label_names.txt',
             similarity_threshold=0.6):
```

**What to implement (Part A):**
1. Add `ros_node=None` as a fourth parameter to `__init__`

**Reasoning:**
- Default `None` means ROS is optional
- When called without `ros_node`, everything works as before
- When called with a `Node` object, publishing is enabled

**Part B:** Find the TODO 25 comment block (after the emotion smoother initialization):

```python
self.emotion_smoother = EmotionSmoother(5, 8)

# TODO 25: Create ROS 2 publishers
```

**What to implement (Part B):**
1. Store the node: `self.ros_node = ros_node`
2. Check if ros_node is not None (guard clause)
3. Inside the guard, create 3 publishers on the node:
   - `/face/identity` for identity results (String, queue_size 10)
   - `/face/emotion` for emotion results (String, queue_size 10)
   - `/face/position` for face position (String, queue_size 10)
4. Store each publisher as `self.identity_pub`, `self.emotion_pub`, `self.position_pub`

**Reasoning:**
- Three topics because the robot needs three pieces of info: WHO, HOW THEY FEEL, WHERE
- `queue_size=10` buffers up to 10 messages if a subscriber is slow
- The guard clause (`if self.ros_node:`) makes publishers conditional

**Hint:** The ROS 2 API for creating a publisher is:
```python
node.create_publisher(MessageType, '/topic/name', queue_size)
```

**Test your implementation:**
```bash
# This should still work without ROS (no ros_node passed):
python3 core/face_recognizer.py
# Expected: Webcam runs normally, no ROS errors
```

---

## Phase 2: Publishing Results (30-45 minutes)

### Concept: Where to Publish

You publish at the point in the code where **all results are ready** for a tracked face. In `run_webcam3`, that's right after the label is drawn on screen:

```python
# Identity is known:     tr["name"], tr["sim"]
# Emotion is known:      emotion_label
# Position is known:     dx, dy, dw, dh (bounding box)
# Label is drawn:        cv2.putText(...)
#
# → This is the right place to also publish via ROS
```

Think of it as adding a second output alongside `cv2.putText`. The screen gets the visual, and ROS gets the data.

### Implementation

**File:** `core/face_recognizer.py`

#### TODO 26: Publish Results to ROS 2 Topics

Navigate to `run_webcam3` and find:

```python
cv2.putText(frame, label, (dx, dy-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# TODO 26: Publish results to ROS 2 topics
```

**What to implement:**
1. Check if `self.ros_node` exists (guard clause)
2. Create a `String` message for identity:
   - Set `msg.data` to a pipe-delimited string: `f'{tr["name"]}|{tr.get("sim", 0):.2f}'`
   - Publish it on `self.identity_pub`
3. Create a `String` message for emotion:
   - Set `msg.data` to the emotion label
   - Publish it on `self.emotion_pub`
4. Create a `String` message for position:
   - Compute face center: `cx = dx + dw // 2`, `cy = dy + dh // 2`
   - Set `msg.data` to: `f'{cx}|{cy}|{dw}|{dh}'`
   - Publish it on `self.position_pub`

**Reasoning:**
- **Identity format `"ben|0.85"`:** The receiving node splits on `|` to get name and confidence separately
- **Emotion format `"Happiness"`:** Simple label string, easy to parse
- **Position format `"320|200|80|100"`:** Center x, center y, width, height of the face bounding box. The navigation node compares the center x to the frame center (e.g., 320 for a 640-wide frame) to decide "turn left" or "turn right"
- **Guard clause:** Without it, the code would crash on `self.identity_pub` when ROS isn't initialized

**Hint:** In ROS 2, you can't just publish a raw string. You must wrap it:
```python
msg = String()
msg.data = "your text here"
publisher.publish(msg)
```

**Common mistake:** Reusing the same `msg` variable for all three publishes. Each publish should use a fresh `String()` object.

#### TODO 27: Initialize ROS 2 Node

Navigate to the `if __name__ == '__main__':` block at the bottom of the file:

```python
if __name__ == '__main__':
    # TODO 27: Initialize ROS 2 and create node
```

**What to implement:**
1. Call `rclpy.init()` to connect to the ROS 2 system
2. Create a node: `ros_node = Node('face_recognition_node')`
3. Pass `ros_node=ros_node` to the `FaceRecognizer` constructor
4. Add a `finally` block that calls `ros_node.destroy_node()` and `rclpy.shutdown()`

**Reasoning:**
- `rclpy.init()` is like logging into the message board. Must happen before any ROS calls.
- `Node('face_recognition_node')` gives your program a name visible to other nodes.
- Passing the node to `FaceRecognizer` enables all the publishers you created in TODO 25.
- Cleanup in `finally` ensures ROS disconnects cleanly even if the program crashes or you press ESC.

**Hint:** The `finally` block runs no matter what -- even after exceptions. Structure it like:
```python
try:
    # ... existing code ...
except ...:
    # ... existing error handling ...
finally:
    ros_node.destroy_node()
    rclpy.shutdown()
```

**Test your implementation:**

Terminal 1 (run your code):
```bash
source /opt/ros/humble/setup.bash
python3 core/face_recognizer.py
```

Terminal 2 (check topics exist):
```bash
source /opt/ros/humble/setup.bash
ros2 topic list
# Should include:
#   /face/identity
#   /face/emotion
#   /face/position
```

Terminal 3 (watch live messages):
```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /face/identity
# Should show messages like:
#   data: 'ben|0.85'
#   ---
#   data: 'Unknown|0.32'
#   ---
```

---

## Testing & Debugging

### Test 1: Topics Exist

```bash
ros2 topic list
# Should show /face/identity, /face/emotion, /face/position
```

If topics don't appear:
- Make sure you sourced ROS: `source /opt/ros/humble/setup.bash`
- Check that `rclpy.init()` was called before creating the node
- Verify the node was passed to `FaceRecognizer`

### Test 2: Messages Are Flowing

```bash
ros2 topic echo /face/identity
# Should show name|similarity every frame a face is tracked
```

```bash
ros2 topic echo /face/emotion
# Should show emotion labels like Happiness, Neutral, etc.
```

```bash
ros2 topic echo /face/position
# Should show cx|cy|w|h coordinates
```

If no messages appear:
- Check that `self.ros_node` is not `None` in the publisher code
- Make sure `TODO 26` is inside the tracked faces loop, after `cv2.putText`
- Verify a face is actually being detected (check the webcam window)

### Test 3: Message Rate

```bash
ros2 topic hz /face/identity
# Should show average rate matching your FPS (e.g., ~15-30 Hz)
```

If rate is too low, the publishing itself is not the bottleneck -- your face detection/recognition is.

### Test 4: Code Still Works Without ROS

```bash
# Without sourcing ROS, the code should still run:
python3 core/face_recognizer.py
# Expected: Works normally, just no ROS publishing
```

If it crashes without ROS, check that:
- `ros_node=None` is the default in `__init__`
- All publish calls are inside `if self.ros_node:` guards

### Common Issues

**Issue 1: "ModuleNotFoundError: No module named 'rclpy'"**
- You haven't sourced ROS 2: run `source /opt/ros/humble/setup.bash`
- Or ROS 2 Humble is not installed on this machine

**Issue 2: "Topics exist but no messages"**
- Face isn't being detected (nothing to publish)
- The publish code is outside the tracked faces loop
- The `if self.ros_node:` guard is preventing publishing (check `ros_node` was passed)

**Issue 3: "rclpy.init() error: context already initialized"**
- `rclpy.init()` was called twice. Make sure it's only called once in `__main__`.

**Issue 4: "Node not spinning / subscribers don't receive"**
- Your webcam loop (`while True`) IS the spin. ROS 2 publishers don't need `rclpy.spin()` -- they push messages immediately on `.publish()`. Subscribers in other nodes handle their own spinning.

---

## What You've Learned

### Technical Skills
- Initializing a ROS 2 node with `rclpy`
- Creating publishers with `create_publisher()`
- Publishing `String` messages to topics
- Making ROS integration conditional (optional dependency)

### Concepts
- Inter-process communication and why it's needed on robots
- Nodes, topics, publishers, and subscribers
- The publish-subscribe pattern
- Why separate processes can't share variables

### Industry Practices
- Optional dependencies (code works with and without ROS)
- Pipe-delimited message formats for simplicity
- Guard clauses to prevent crashes when features are disabled
- Clean shutdown with `finally` blocks

---

## Summary

You've added ROS 2 publishing to the face recognition system:

1. **Phase 1:** Imported ROS 2 libraries and created 3 publishers (identity, emotion, position)
2. **Phase 2:** Published results inside the webcam loop and initialized the ROS 2 node

**Key insight:** ROS integration doesn't require rewriting your code. It's a thin output layer -- a few `publish()` calls added alongside the existing `cv2.putText` calls. The same results go to the screen AND to the robot.

**What changed:** 4 TODOs, ~15 lines of new code across one file.

**What didn't change:** Everything else -- detection, recognition, emotion, smoothing, tracking, display.

---

## Appendix: ROS 2 Quick Reference

### Common Commands

```bash
# Source ROS 2 (do this in every new terminal)
source /opt/ros/humble/setup.bash

# List all running nodes
ros2 node list

# List all active topics
ros2 topic list

# Watch messages on a topic
ros2 topic echo /face/identity

# Check message rate
ros2 topic hz /face/identity

# See topic info (type, publishers, subscribers)
ros2 topic info /face/identity
```

### ROS 2 Python Cheat Sheet

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Initialize
rclpy.init()
node = Node('my_node_name')

# Create publisher
pub = node.create_publisher(String, '/my/topic', 10)

# Publish a message
msg = String()
msg.data = 'hello'
pub.publish(msg)

# Cleanup
node.destroy_node()
rclpy.shutdown()
```

### What Other Nodes See

When your face recognition node is running, other teams can subscribe:

```python
# In another node (e.g., arm controller):
def on_identity(msg):
    name, confidence = msg.data.split('|')
    if name == 'ben':
        wave_arm()

node.create_subscription(String, '/face/identity', on_identity, 10)
```

---

**Next:** Start with Phase 1 -- open `core/face_recognizer.py` and implement TODO 24!
