# Traffic counter

## About

This script is for real-time object tracking and analysis. The script makes use of the following libraries:

- OpenCV
- pyTorch/YOLOv5
- SORT

The script tracks entities such as people, bicycles, and cars using YOLOv5. The entities are tracked using the Sort tracking algorithm. The following information is stored for each entity:

- History: The history of the entity's position over time.
- Velocity History: The history of the entity's velocity over time.
- Summary: Statistics about the entity's detections.

The script captures screenshots of the screen and uses them for tracking the entities. The screen dimensions and the location from where the screenshot is taken are defined by the constants WIDTH, HEIGHT, TOP, and LEFT. The script converts the screenshot from BGR to RGB and then performs object detection using YOLOv5. The detections are then fed into the Sort tracker to track the entities.

The script calculates the velocity of each entity by using the position history. The velocity calculation is performed every time a new detection is added to the history. The velocity history of each entity is stored in a list.

## Getting Started

    To get a local copy of YOLOv5, follow these steps:
    1. Clone the repository: Use the command git clone https://github.com/ultralytics/yolov5.git in your terminal or command prompt to download a local copy of YOLOv5.
    2. Install the required dependencies: Make sure you have installed Python 3.7 or higher, pip, and CUDA 10.2 or higher. You can use the command pip install -r requirements.txt to install all required dependencies.
    3. Build the Cython extensions: Use the command python setup.py build_ext --inplace to build the Cython extensions.
    4. Run the tests: Use the command python test.py to run the tests and check if everything is working correctly.
    5. Use custom model wrapper: copy `./customWrapper.py` to the directory of the `yolov5-master` repo.

    Change PATH and SOURCE constants in `app.py` to suit the requirements.
    Run `app.py` with python3 to begin.

## Status

The script has not been completed yet. To-do items are mentioned in the code comments. These include:

- Plotting the count with time
- Finding the maximum number of entities at one time
- Finding the average velocity of the entities
- Finding the number of entities at different times

Note that this script is a work in progress and may contain bugs or unfinished code. Use it at your own risk.