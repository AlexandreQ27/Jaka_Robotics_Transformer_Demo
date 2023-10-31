import os  
import cv2  
import numpy as np
def undistort_images(input_dir, output_dir, camera_matrix, dist_coeff):  
    """  
    对输入目录中的每个图像进行去畸变处理，并将结果保存到输出目录。  
      
    参数：  
    input_dir：输入目录路径。  
    output_dir：输出目录路径。  
    camera_matrix：相机矩阵。  
    dist_coeff：畸变系数。  
    """  
    # 获取输入目录中的所有文件名  
    file_names = os.listdir(input_dir)  
      
    # 对每个文件进行处理  
    for file_name in file_names:  
        # 读取图像  
        img_path = os.path.join(input_dir, file_name)  
        img = cv2.imread(img_path)  
          
        # 去畸变处理  
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeff)  
          
        # 保存去畸变后的图像  
        output_path = os.path.join(output_dir, file_name)  
        cv2.imwrite(output_path, undistorted_img)  
        print(f"Processed {file_name} and saved as {output_path}")  
  
# 设置输入和输出目录  
input_dir = "/home/qyb/simplified_eye_hand_calibration/build/data/"  
output_dir = "/home/qyb/Desktop/color_images/"  
  
# 设置相机矩阵和畸变系数，这些值通常是通过相机标定得到的  
camera_matrix=np.array([[607.2513784473566, 0, 312.0057806991867],
 [0, 607.5275453454146, 238.6361143847473],
 [0, 0, 1]])
dist_coeff=np.float32([-0.02220346593665167,
 1.294554632487039,
 0.006218227086377285,
 -0.009092912927694383,
 -4.902303897433339])
 
  
# 调用函数进行去畸变处理  
undistort_images(input_dir, output_dir, camera_matrix, dist_coeff)
