# yolo-reloaded
This project is a complete rewrite of the yolo/darknet project.

The original project is YOLO: Real-Time Object Detection by Joseph Redmon, web site @ https://pjreddie.com/darknet/yolo/

The purpose of yolo-reloaded is a complete rewrite of the original code in order ease understanding of what it does and ease its modification by people new to the project. This is done using C++ 14 (probably 17 when available).

You may retrieve the data and cfg directories on the https://pjreddie.com/darknet/yolo/ site (ok, yet another todo, maybe add it add a submodule would be easier)

As the amount of work is quite huge, the project will try to follow theses steps:

* have a simple example to work (for instance ./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg)
* have OpenCL to be used (if you feell like writing the cuda code, you are welcoe, actually I do not have a nvidia card for testing...)
* have live detection on webcam work
* have network training work

Obviously, this may change depending on the impediments/needs encountered during development.

Feel free to email me about any remarks and to propose patches (even a single line patch can change everything)

