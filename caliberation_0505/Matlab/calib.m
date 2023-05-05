clear;
close all;
dbstop error; 
clc;

base_dir  = 'E:/KITTI/2011_09_26/2011_09_26_drive_0009_sync'; % 图片目录
calib_dir = 'E:/KITTI/2011_09_26_calib/2011_09_26'; % 相机参数目录

cam       = 2; % 第3个摄像头
frame     = 5; % 第0帧(第一张图片)

calib1 = loadCalibrationCamToCam(fullfile(calib_dir,'calib_cam_to_cam.txt'));
Tr_velo_to_cam = loadCalibrationRigid(fullfile(calib_dir,'calib_velo_to_cam.txt'));

% 计算点云到图像平面的投影矩阵
R_cam_to_rect = eye(4);
R_cam_to_rect(1:3,1:3) = calib1.R_rect{1}; % R_rect：纠正旋转使图像平面共面
P_velo_to_img = calib1.P_rect{cam+1}*R_cam_to_rect*Tr_velo_to_cam; % 内外参数 P_rect：矫正后的投影矩阵

img = imread(sprintf('%s/image_%02d/data/%010d.png', base_dir, cam, frame));
imshow(img); hold on;

fid = fopen(sprintf('%s/velodyne_points/data/%010d.bin',base_dir,frame),'rb');
velo = fread(fid,[4 inf],'single')';
velo = velo(1:1:end,:); % 显示速度每5点移除一次
fclose(fid);

% 删除图像平面后面的所有点（近似值）
idx = velo(:,1)<5; 
velo(idx,:) = [];

% 投影到图像平面（排除亮度）
velo_img = project(velo(:,1:3),P_velo_to_img);

% 画点
cols = jet;
for i=1:size(velo_img,1)
  col_idx = round(256*5/velo(i,1));
  plot(velo_img(i,1),velo_img(i,2),'o','LineWidth',2,'MarkerSize',0.5,'Color',cols(col_idx,:));
end