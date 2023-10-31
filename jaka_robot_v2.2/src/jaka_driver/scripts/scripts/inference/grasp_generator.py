import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData
from utils.dataset_processing.grasp import detect_grasps
from utils.visualisation.plot import plot_grasp


class GraspGenerator:
    def __init__(self, saved_model_path, cam_id, visualize=False):
        self.saved_model_path = saved_model_path
        self.camera = RealSenseCamera(device_id=cam_id)

        self.saved_model_path = saved_model_path
        self.model = None
        self.device = None

        self.cam_data = CameraData(include_depth=True, include_rgb=True)

        # Connect to camera
        self.camera.connect()

        # Load camera pose and depth scale (from running calibration)
        #self.cam_pose = np.loadtxt('saved_data/camera_pose.txt', delimiter=' ')
        #self.cam_depth_scale = np.loadtxt('saved_data/camera_depth_scale.txt', delimiter=' ')

        homedir = os.path.join(os.path.expanduser('~'), "grasp-comms")
        self.grasp_request = os.path.join(homedir, "grasp_request.npy")
        self.grasp_available = os.path.join(homedir, "grasp_available.npy")
        self.grasp_pose = os.path.join(homedir, "grasp_pose.npy")

        if visualize:
            self.fig = plt.figure(figsize=(10, 10))
        else:
            self.fig = None

    def load_model(self):
        print('Loading model... ')
        self.model = torch.load(self.saved_model_path,map_location=torch.device('cpu'))
        # Get the compute device
        self.device = get_device(force_cpu=False)

    def generate(self):
        # Get RGB-D image from camera
        image_bundle = self.camera.get_image_bundle()
        rgb = image_bundle['rgb']
        depth = image_bundle['aligned_depth']
        x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

        # Predict the grasp pose using the saved model
        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.model.predict(xc)
        #print(pred)
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
        grasps = detect_grasps(q_img, ang_img, width_img)
        rgb_img = np.transpose(rgb_img, (1, 2, 0))
        #ax = fig.add_subplot(2, 2, 1)
        # rgb_img = np.transpose(rgb_img, (1, 2, 0))
        # Get grasp position from model output
        # rgb_img*=255.0
        print(rgb)
        cv2.imshow("rgb_img",rgb)
        k = cv2.waitKey(0) # waitKey代表读取键盘的输入，0代表一直等待
        if k ==27:     # 键盘上Esc键的键值
            cv2.destroyAllWindows() 
        

        

    def run(self):
        while True:
            #if np.load(self.grasp_request):
            # Get RGB-D image from camera
            image_bundle = self.camera.get_image_bundle()
            rgb = image_bundle['rgb']
            depth = image_bundle['aligned_depth']
            x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)

            # Predict the grasp pose using the saved model
            with torch.no_grad():
                xc = x.to(self.device)
                pred = self.model.predict(xc)
            #print(pred)

            q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'])
            grasps = detect_grasps(q_img, ang_img, width_img)
            print(rgb_img.shape)
            rgb_img = np.transpose(rgb_img, (1, 2, 0))
            for g in grasps:
                # point_1=g.as_gr.points[0]
                # point_2=g.as_gr.points[1]
                # point_3=g.as_gr.points[2]
                # point_4=g.as_gr.points[3]
                print(g.center)
                #pt1=int([g.as_gr.points[0][0],g.as_gr.points[0][1]]),pt2=int([g.as_gr.points[1][0],g.as_gr.points[1][1]])
                print(self.cam_data.top_left)
                draw_0 = cv2.rectangle(rgb_img, pt1=[int(g.as_gr.points[4][0]),int(g.as_gr.points[4][1])],pt2=[int(g.as_gr.points[5][0]),int(g.as_gr.points[5][1])], color=(255, 0, 0), thickness=2)
                pos_z = depth[grasps[0].center[0] + self.cam_data.top_left[0], grasps[0].center[1] + self.cam_data.top_left[1]] 
                pos_x = np.multiply(grasps[0].center[1] + self.cam_data.top_left[1] - self.camera.intrinsics.ppx,
                            pos_z / self.camera.intrinsics.fx)
                pos_y = np.multiply(grasps[0].center[0] + self.cam_data.top_left[0] - self.camera.intrinsics.ppy,
                            pos_z / self.camera.intrinsics.fy)
                target = np.asarray([pos_x, pos_y, pos_z])
                target.shape = (3, 1)
                print('target: ', target)
            #ax = fig.add_subplot(2, 2, 1)
            # rgb_img = np.transpose(rgb_img, (1, 2, 0))
            # rgb_img*=255.0
            #print(rgb)
            cv2.imshow("rgb_img",draw_0)
            if cv2.waitKey(10) & 0xFF == 27 :
                break 
