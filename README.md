# Simple Offside Detector

A simple tool to recognize football/soccer offside in still images.

## How it works

1. Mask out out-of-field areas.
2. Run player detection using *YOLOv3 weights on Darknet Neural Network*.
3. Classify players to 4 groups (2 teams, GK and referees) based on *color histogram*.
4. User selects attacking team & defending team. GK is counted in defending team.
5. Calculate *vanishing point* (VP).
6. Draw defending line crossing the leftmost point of 2nd lowest player of the defending team and VP.
7. Draw attacking line crossing the leftmost point of the highest player of the attacking team and VP.
8. If attacking line is more to the left than the defending line => OFFSIDE. \
   Else => NOT OFFSIDE.

## Running the tool

1. Make sure you have Python 3.9+ installed.

2. The first time running, please install darknet, YOLOv3 weights and requirements:

   ```(shell)
   git clone https://github.com/pjreddie/darknet
   cd darknet
   make
   wget https://pjreddie.com/media/files/yolov3.weights
   cd ..
   pip install requirement.txt
   ```

3. Run on example data:

   ```(shell)
   python main.py
   ```
