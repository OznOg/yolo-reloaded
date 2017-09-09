# yolo-reloaded
This project is a complete rewrite of the yolo/darknet project.

The original project is YOLO: Real-Time Object Detection by Joseph Redmon, web site @ https://pjreddie.com/darknet/yolo/

The purpose of yolo-reloaded is a complete rewrite of the original code in order ease understanding of what it does and ease its modification by people new to the project. This is done using C++ 14 (probably 17 when available).

As the amount of work is quite huge, the project will try to follow theses steps:

* have a simple example to work (for instance ./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg)
* have CUDA used
* have live detection on webcam work
* have network training work

Obviously, this may change depending on the impediments/needs encountered during development.
