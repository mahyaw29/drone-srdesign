import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

camera_calibration_parameters = 'calibration_chessboard.yaml'

# side length of aruco tag in meters
aruco_marker_length = 0.0785

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

def main():
    cv_file = cv2.FileStorage(camera_calibration_parameters, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode('K').mat()
    dst = cv_file.getNode('D').mat()
    cv_file.release()
        
    parameters = cv2.aruco.DetectorParameters_create()
    
    cap = cv2.VideoCapture(0)
    
    while(True):
        ret, frame = cap.read()
        
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters, cameraMatrix=mtx, distCoeff=dst)
        
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)
            rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_length, mtx, dst)
        
            for i, marker_id in enumerate(marker_ids):
                transform_translation_x = tvecs[i][0][0]
                transform_translation_y = tvecs[i][0][1]
                transform_translation_z = tvecs[i][0][2]
                
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
                print("marker id: {}".format(i))
                print("transform_translation_x: {}".format(transform_translation_x))
                print("tranform_translation_y: {}".format(transform_translation_y))
                print("transform_translation_z: {}".format(transform_translation_z))
                print("roll_x: {}".format(roll_x))
                print("pitch_y: {}".format(pitch_y))
                print("yaw_z: {}".format(yaw_z))
                print()
                
                cv2.aruco.drawAxis(frame, mtx, dst, rvecs[i], tvecs[i], 0.04)
            
        cv2.imshow('Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()