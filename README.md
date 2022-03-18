# TelloProjects
Python script to control [DJI Tello](https://store.dji.com/product/tello). This repo utilizes the DJI Tello SDK. Frames are recorded by the drone, sent to a computer to be processed and put through deep learning models. The output will be processed and map into orders to the active drone through wifi. The camera on the drone runs on 30FPS.

## 0. Installation:
```
git clone https://github.com/BatmanofZuhandArrgh/TelloProjects.git
conda create --name tello
conda activate tello
pip install -r requirements.txt
```
Turn on the drone and connect to a Linux/Windows machine with wifi (macOS is not tested)
## 1. User controls:
Run
```
python features/user_controlled.py
```
Press t to take off, l or q to land.
Use arrow keys, w,a,s,d for other control command.

## 2. Hand pose controls:
Run 
```
python features/handpose_controlled.py
```
[Mediapipe handpose landmark detection API](https://google.github.io/mediapipe/solutions/hands.html) was used to process frames.
Press t to take off, l or q to land.
By default, the direction of the index finger is used to map up, down, left, right commands.
Rock and roll sign is used to command a random flip.
Mockingjay sign is used to command land.

Demo link: 

## 3. Face tracking controls:
Run 
```
python features/facetracking_controlled.py
```
[timesler's mtcnn model](https://github.com/timesler/facenet-pytorch/blob/master/models/mtcnn.py) was used to detect faces.
Press t to take off, l or q to land. The drone would divide the frame into 9 unequal rectangular parts. Once the one biggest face detected on the 8 outside parts, the drone will move to center the face again.

Demo link: 
