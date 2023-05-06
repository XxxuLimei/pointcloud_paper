import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
import cv2

base_dir = '/home/xilm/kitti/raw_data/2011_09_26_drive_0084_sync/2011_09_26/2011_09_26_drive_0084_sync/'
calib_dir = '/home/xilm/kitti/raw_data/2011_09_26_drive_0084_sync/2011_09_26/'

cam = 2
frame = 5


def loadCalibrationCamToCam(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    calib = {}
    obj = lines[1].strip().split(' ')[1:]
    calib['corner_dist'] = obj

    for i in range(4):
        obj = lines[2+i*8].strip().split(' ')[1:]
        S_ = np.array(obj, dtype=np.float32).reshape(1, 2)
        obj = lines[3+i*8].strip().split(' ')[1:]
        K_ = np.array(obj, dtype=np.float32).reshape(3, 3)
        obj = lines[4+i*8].strip().split(' ')[1:]
        D_ = np.array(obj, dtype=np.float32).reshape(1, 5)
        obj = lines[5+i*8].strip().split(' ')[1:]
        R_ = np.array(obj, dtype=np.float32).reshape(3, 3)
        obj = lines[6+i*8].strip().split(' ')[1:]
        T_ = np.array(obj, dtype=np.float32).reshape(3, 1)
        obj = lines[7+i*8].strip().split(' ')[1:]
        S_rect_ = np.array(obj, dtype=np.float32).reshape(1, 2)
        obj = lines[8+i*8].strip().split(' ')[1:]
        R_rect_ = np.array(obj, dtype=np.float32).reshape(3, 3)
        obj = lines[9+i*8].strip().split(' ')[1:]
        P_rect_ = np.array(obj, dtype=np.float32).reshape(3, 4)

        calib['S_%02d' % i] = S_
        calib['K_%02d' % i] = K_
        calib['D_%02d' % i] = D_
        calib['R_%02d' % i] = R_
        calib['T_%02d' % i] = T_
        calib['S_rect_%02d' % i] = S_rect_
        calib['R_rect_%02d' % i] = R_rect_
        calib['P_rect_%02d' % i] = P_rect_
    return calib


def loadCalibrationRigid(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[1].strip().split(' ')[1:]
    R = np.array(obj, dtype=np.float32).reshape(3, 3)
    obj = lines[2].strip().split(' ')[1:]
    T = np.array(obj, dtype=np.float32).reshape(3, 1)
    Tr = np.concatenate([R, T], axis=1)
    Tr = np.concatenate((Tr, np.array([[0, 0, 0, 1]])), axis=0)
    return Tr


calib = loadCalibrationCamToCam(calib_dir + 'calib_cam_to_cam.txt')
Tr_velo_to_cam = loadCalibrationRigid(calib_dir + 'calib_velo_to_cam.txt')

def Tr_velo_to_img(calib, Tr_velo_to_cam, cam):
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = calib['R_rect_00'] # because velodyne is trans to camera 00
    P_velo_to_img = np.matmul(R_cam_to_rect, Tr_velo_to_cam)
    P_velo_to_img = np.matmul(calib['P_rect_%02d'%cam], P_velo_to_img)
    return P_velo_to_img

Pro = Tr_velo_to_img(calib, Tr_velo_to_cam, cam)

img = Image.open(base_dir + os.path.join('image_%02d'%cam, 'data', '%010d.png'%frame))

lidar_file = base_dir + 'velodyne_points/' + 'data/' + ('%010d.bin' % frame)
velo_points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

idx = np.where(velo_points[:,0] >= 5)
velo_points = velo_points[idx]

def project(velo_points, Pro):
    dim_norm = Pro.shape[0]
    dim_proj = Pro.shape[1]

    if velo_points.shape[1]<dim_proj:
        velo_points = np.concatenate([velo_points, np.ones((velo_points.shape[0], 1))], axis=1)
    velo_points = np.transpose(np.matmul(Pro, np.transpose(velo_points)))
    return velo_points[:, 0:dim_norm-1] / np.matmul(velo_points[:, -1:], np.ones((1, dim_norm-1)))

velo_img = project(velo_points[:, 0:3], Pro)

imagedraw = ImageDraw.Draw(img)
for i in range(velo_img.shape[0]):
    col = np.floor(255 / np.max(velo_img[:, 0], 0) * velo_img[i, 0]*5)
    x = velo_img[i, 0]
    y = velo_img[i, 1]
    imagedraw.point((x, y),(int(col), 0, 0))
img.show()
