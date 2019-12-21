# Vehicle-Detection-using-Intel-OpenVINO

An implementation of a cost-effective, real-time vehicle detection application on the edge using the OpenVINO toolkit.

The pre-trained model (vehicle-detection-adas-0002) is extracted from the Intel Open Model Zoo.This is a vehicle detection network based on an SSD framework with tuned MobileNet v1 as a feature extractor. 

*Sample Usage:*
`python app.py -m ./Model/vehicle-detection-adas-0002.xml -ct 0.6 -c BLUE -i ./Input/driving_berlin.mp4`

Here, **-m** specifies the path to the model file, **-ct** specifies the confidence threshold, **-c** is the desired color for the boxes drawn on the output video and **-i** specifies the path to the input file.

This project has been inspired from the Udacity Intel Edge AI Foundation Course.
