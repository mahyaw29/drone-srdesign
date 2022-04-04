# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 15:01:39 2022

@author: Mahya
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import pandas as pd

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

camera_calibration_parameters = 'calibration_chessboard.yaml'

# side length of aruco tag in meters
aruco_marker_length = 0.1

def euler_from_quaternion(x, y, z, w):
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z # in radians

def calculateDistance(droneTvec, markerTvecList):
    d_list = []
    for i in range(0, len(markerTvecList)):
        secondTvec = markerTvecList[i]
        (droneTvec - secondTvec).any()
        distance = np.linalg.norm(droneTvec - secondTvec)
        d_list.append(distance)
    return d_list

def saveToCSV(markerIDList, markerDistance):
    markerDistance = list(markerDistance)
    dict = {k:v for (k,v) in zip(markerIDList, markerDistance)}
    print(dict)
    df = pd.DataFrame(dict)
    df.to_csv('eval.csv')
    

def main():
    cv_file = cv2.FileStorage(camera_calibration_parameters, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
        
    parameters = cv2.aruco.DetectorParameters_create()
    
    cap = cv2.VideoCapture(0)
    drone_marker_id = 24
    droneMarkerCalibrated = False
    
    # set how many obstacle markers used here
    markerDistances = np.empty((2,0), float)
    markerTvecList = []
    markerIDList = []
    
    while(True):
        ret, frame = cap.read()
        
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dst)
        
        if marker_ids is not None:
            del markerTvecList[:]
            del markerIDList[:]
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, mtx, dst)
        
            for i in range(0, len(marker_ids)):
                if marker_ids[i] == drone_marker_id:
                    #firstRvec = rvecs[i]
                    droneTvec = tvecs[i]
                    droneMarkerCalibrated = True
                    #firstMarkerCorners = corners[i]
                else:
                    markerTvecList.append(tvecs[i])
                    markerIDList.append(int(marker_ids[i]))
                    
            if len(markerTvecList) > 0 and droneMarkerCalibrated == True:
                calc_distance = calculateDistance(droneTvec, markerTvecList)
                calc_distance_t = np.array([calc_distance]).transpose()
                markerDistances = np.append(markerDistances, calc_distance_t, axis=1)
                #for i in range(0, len(markerTvecList)):
                    #print(markerIDList[i], ': ', calc_distance[i])
                
                
                #distance_over_time.append(distance)
                #id15_tvec.append(firstTvec)
                #id24_tvec.append(secondTvec)
                #print("marker 15 TVEC: {}".format(firstTvec))
                #print("marker 24 TVEC: {}".format(secondTvec))
                #print("distance between markers: ", distance)
                #print()
                
                #transform_translation_x = tvecs[i][0][0]
                #transform_translation_y = tvecs[i][0][1]
                #transform_translation_z = tvecs[i][0][2]
                
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()
                
                transform_rotation_x = quat[0]
                transform_rotation_y = quat[1]
                transform_rotation_z = quat[2]
                tranform_rotation_w = quat[3]
                
                roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, transform_rotation_y, transform_rotation_z, tranform_rotation_w)
                
                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)
                
                #print("marker id: {}".format(i))
                #print("transform_translation_x: {}".format(transform_translation_x))
                #print("tranform_translation_y: {}".format(transform_translation_y))
                #print("transform_translation_z: {}".format(transform_translation_z))
                #print("roll_x: {}".format(roll_x))
                #print("pitch_y: {}".format(pitch_y))
                #print("yaw_z: {}".format(yaw_z))
                #print()
                
                cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.04)
            
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            #print(markerDistances)
            saveToCSV(markerIDList, markerDistances)
            break
            
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()