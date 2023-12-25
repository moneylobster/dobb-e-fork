#! /usr/bin/env python

import message_filters
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from control_msgs.msg import JointTrajectoryControllerState

import numpy as np
import json
import os 
from datetime import datetime


# get the date in YYYY-MM-DD-HH-MM-SS format
datename = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
lasttime=0
iter = 0
jsonpath = f"data/{datename}.json"
jsonfile = None
imgarr=np.empty((480,640,3,1), dtype=np.uint8)
deptharr=np.empty((480,640,1,1), dtype=np.int16)

def cleanup():
    global jsonfile, imgarr, deptharr
    jsonfile.close()
    np.savez(f'data/{datename}.npz',img=imgarr,depth=deptharr)


def callback(image, depth, vel, torso, gripper):
    global lasttime, jsonfile, iter, imgarr, deptharr

    thistime=rospy.get_rostime()
    rospy.loginfo(f"Received a synced message! with {1/(thistime.nsecs-lasttime.nsecs) * 1e9} Hz")
    lasttime=thistime
    #save data
    '''
    print(f"""Data:
          Img: {image.height} x {image.width}
          Depth: {depth.height} x {depth.width} {depth.encoding}
          Vel: lin {vel.linear.x} ang {vel.angular.z}
          Torso: {torso.actual.positions[0]}
          Gripper: {gripper.actual.positions[0]}""")
    '''
    data = {f"{iter}":{"vw":[vel.linear.x,vel.angular.z],"torso":torso.actual.positions[0],"gripper":gripper.actual.positions[0]}}
    json.dump(data,jsonfile)
    iter += 1
    npimg=np.frombuffer(image.data, np.uint8).reshape(480,640,3,1)
    npdepth=np.frombuffer(depth.data, np.int16).reshape(480,640,1,1)
    
    #print(f"imgarr {imgarr.shape} npimg {npimg.shape}")
    imgarr=np.concatenate((imgarr, npimg), axis=-1)
    deptharr=np.concatenate((deptharr, npdepth), axis=-1)


def listener():
    
    global lasttime, jsonfile
    if not os.path.exists(os.path.dirname(jsonpath)):
        os.mkdir(os.path.dirname(jsonpath))
    jsonfile = open(jsonpath,'a') 

    rospy.on_shutdown(cleanup)   # close the file during shutdown


    rospy.init_node('listener', anonymous=True)
    lasttime=rospy.get_rostime()
    image_sub = message_filters.Subscriber("/xtion/rgb/image_rect_color", Image)
    depth_sub = message_filters.Subscriber("/xtion/depth/image_raw", Image)
    vel_sub = message_filters.Subscriber("/mobile_base_controller/cmd_vel", Twist)
    torso_sub = message_filters.Subscriber("/torso_controller/state", JointTrajectoryControllerState)
    gripper_sub = message_filters.Subscriber("/gripper_controller/state", JointTrajectoryControllerState)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub, vel_sub, torso_sub, gripper_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
    


