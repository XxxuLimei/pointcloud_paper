# pointcloud_paper
Some papers about pointcloud 3D detection  
## 0505:  
1. VoxelNet: [Pytorch实现](https://github.com/skyhehe123/VoxelNet-pytorch)  
- 学习了点云与图像如何进行对齐，并使用Matlab code进行了数据校准，将该代码使用Python进行实现。  
- Matlab绘制结果  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/figure/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230505203558.png)  
- Python绘制结果  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/caliberation_0505/Python/tmp2avsrejr.PNG)  
## 0506:  
1. 发现上面那个仓库的代码在编译nms model的时候失败了，于是换了一个[仓库](https://github.com/RPFey/voxelnet_pytorch)  
- 首先Pip detectron2;  
- 发现还是需要编译box_overlaps：  
```
(base) xilm@xilm-MS-7D17:~/fuxian/voxelnet_pytorch-master$ python3 setup.py build_ext --inplace
running build_ext
building 'box_overlaps' extension
creating build
creating build/temp.linux-x86_64-3.9
gcc -pthread -B /home/xilm/anaconda3/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -O2 -Wall -fPIC -O2 -isystem /home/xilm/anaconda3/include -I/home/xilm/anaconda3/include -fPIC -O2 -isystem /home/xilm/anaconda3/include -fPIC -I/home/xilm/anaconda3/lib/python3.9/site-packages/numpy/core/include -I/home/xilm/anaconda3/include/python3.9 -c ./box_overlaps.c -o build/temp.linux-x86_64-3.9/./box_overlaps.o
In file included from /home/xilm/anaconda3/lib/python3.9/site-packages/numpy/core/include/numpy/ndarraytypes.h:1948,
                 from /home/xilm/anaconda3/lib/python3.9/site-packages/numpy/core/include/numpy/ndarrayobject.h:12,
                 from /home/xilm/anaconda3/lib/python3.9/site-packages/numpy/core/include/numpy/arrayobject.h:5,
                 from ./box_overlaps.c:759:
/home/xilm/anaconda3/lib/python3.9/site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h:17:2: warning: #warning "Using deprecated NumPy API, disable it with " "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
   17 | #warning "Using deprecated NumPy API, disable it with " \
      |  ^~~~~~~
gcc -pthread -B /home/xilm/anaconda3/compiler_compat -shared -Wl,-rpath,/home/xilm/anaconda3/lib -Wl,-rpath-link,/home/xilm/anaconda3/lib -L/home/xilm/anaconda3/lib -L/home/xilm/anaconda3/lib -Wl,-rpath,/home/xilm/anaconda3/lib -Wl,-rpath-link,/home/xilm/anaconda3/lib -L/home/xilm/anaconda3/lib build/temp.linux-x86_64-3.9/./box_overlaps.o -o /home/xilm/fuxian/voxelnet_pytorch-master/box_overlaps.cpython-39-x86_64-linux-gnu.so
```  
运行完成后生成`box_overlaps.cpython-39-x86_64-linux-gnu.so`文件。  
- 接着在终端输入`python train.py --index 30 --epoch 30`就可以开始训练了；  
- 突然发现，tensorboard文件最后的数字中，包含端口号信息，比如`events.out.tfevents.1683438487.xilm-MS-7D17.12148.0`就表示端口号是`12148`;  
- 运行后，在tensorboard还可以实时显示pred box与gt box的对比：  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/figure/Screenshot%20from%202023-05-07%2018-22-08.png)  
- 这个训练速度确实是太慢了，跑了三个小时才跑了一个epoch，后面就不准备继续训练了。放了两张训练时的损失下降曲线图。  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/figure/Screenshot%20from%202023-05-07%2018-22-22.png)  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/figure/Screenshot%20from%202023-05-07%2018-22-28.png)  
2. PointNet: [Pytorch实现](https://github.com/fxia22/pointnet.pytorch)  
## 0601：PVRCNN复现及源码阅读  
1. 关于kitti数据集生成的dbinfos信息：  
```
在Kitti数据集中，dbinfos是一个Python列表，每个元素包含了一个数据样本的相关信息。它是用于描述Kitti数据集中每个样本的数据库信息。
每个样本的dbinfos通常包括以下信息：
point_cloud：点云数据的文件路径或点云数据本身。
calib：相机和激光雷达之间的标定信息，包括相机内参、外参和激光雷达的参数等。
image：图像数据的文件路径或图像数据本身。
lidar_idx：对应于点云数据的索引或标识符。
image_idx：对应于图像数据的索引或标识符。
image_shape：图像数据的尺寸，通常是宽度和高度。
num_points_in_gt：Ground Truth目标中的点云数量。
gt_boxes：Ground Truth目标的边界框坐标。
通过访问dbinfos中的元素，可以获取每个数据样本的具体信息，如点云数据路径、图像数据路径、标定信息和目标边界框等。这些信息对于进行目标检测、目标跟踪和传感器融合等任务非常有用。
```  

## 0605:  
1. 使用MMDetection3D框架，尝试运行了SECOND网络，作出了如下修改：  
- 在`configs/_base_/datasets/kitti-3d-3class.py`文件中，修改了lr为0.00125，并将迭代次数修改为2；  
- 打印出配置文件如下：  
```
voxel_size = [0.05, 0.05, 0.1]
model = dict(
    type='VoxelNet',
    voxel_layer=dict(
        max_num_points=5,
        point_cloud_range=[0, -40, -3, 70.4, 40, 1],
        voxel_size=[0.05, 0.05, 0.1],
        max_voxels=(16000, 40000)),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[[0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -0.6, 70.4, 40.0, -0.6],
                    [0, -40.0, -1.78, 70.4, 40.0, -1.78]],
            sizes=[[0.8, 0.6, 1.73], [1.76, 0.6, 1.73], [3.9, 1.6, 1.56]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(
                type='MaxIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.6,
                neg_iou_thr=0.45,
                min_pos_iou=0.45,
                ignore_iof_thr=-1)
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))
dataset_type = 'KittiDataset'
data_root = '/home/xilm/kitti/KITTI/'
class_names = ['Pedestrian', 'Cyclist', 'Car']
point_cloud_range = [0, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
file_client_args = dict(backend='disk')
db_sampler = dict(
    data_root='/home/xilm/kitti/KITTI/',
    info_path='/home/xilm/kitti/KITTI/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
    classes=['Pedestrian', 'Cyclist', 'Car'],
    sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    file_client_args=dict(backend='disk'))
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=dict(backend='disk')),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='/home/xilm/kitti/KITTI/',
            info_path='/home/xilm/kitti/KITTI/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(Car=5, Pedestrian=10, Cyclist=10)),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            file_client_args=dict(backend='disk'))),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0.5],
        global_rot_range=[0.0, 0.0],
        rot_range=[-0.78539816, 0.78539816]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(
        type='ObjectRangeFilter', point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Pedestrian', 'Cyclist', 'Car'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Pedestrian', 'Cyclist', 'Car'],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='/home/xilm/kitti/KITTI/',
            ann_file='/home/xilm/kitti/KITTI/kitti_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='/home/xilm/kitti/KITTI/',
                        info_path=
                        '/home/xilm/kitti/KITTI/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(
                                Car=5, Pedestrian=10, Cyclist=10)),
                        classes=['Pedestrian', 'Cyclist', 'Car'],
                        sample_groups=dict(Car=12, Pedestrian=6, Cyclist=6),
                        points_loader=dict(
                            type='LoadPointsFromFile',
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4,
                            file_client_args=dict(backend='disk')),
                        file_client_args=dict(backend='disk'))),
                dict(
                    type='ObjectNoise',
                    num_try=100,
                    translation_std=[1.0, 1.0, 0.5],
                    global_rot_range=[0.0, 0.0],
                    rot_range=[-0.78539816, 0.78539816]),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['Pedestrian', 'Cyclist', 'Car']),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=['Pedestrian', 'Cyclist', 'Car'],
            test_mode=False,
            box_type_3d='LiDAR',
            file_client_args=dict(backend='disk'))),
    val=dict(
        type='KittiDataset',
        data_root='/home/xilm/kitti/KITTI/',
        ann_file='/home/xilm/kitti/KITTI/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk')),
    test=dict(
        type='KittiDataset',
        data_root='/home/xilm/kitti/KITTI/',
        ann_file='/home/xilm/kitti/KITTI/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -3, 70.4, 40, 1]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Pedestrian', 'Cyclist', 'Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Pedestrian', 'Cyclist', 'Car'],
        test_mode=True,
        box_type_3d='LiDAR',
        file_client_args=dict(backend='disk')))
evaluation = dict(
    interval=1,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Pedestrian', 'Cyclist', 'Car'],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
lr = 0.00125
optimizer = dict(
    type='AdamW', lr=0.00125, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=2)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/hv_second_secfpn_6x8_80e_kitti-3d-3class'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = [0]

2023-06-05 19:13:17,054 - mmdet - INFO - Set random seed to 0, deterministic: False
/home/xilm/mmlab/mmdetection3d-master/mmdet3d/models/dense_heads/anchor3d_head.py:84: UserWarning: dir_offset and dir_limit_offset will be depressed and be incorporated into box coder in the future
  warnings.warn(
2023-06-05 19:13:17,083 - mmdet - INFO - initialize SECOND with init_cfg {'type': 'Kaiming', 'layer': 'Conv2d'}
2023-06-05 19:13:17,104 - mmdet - INFO - initialize SECONDFPN with init_cfg [{'type': 'Kaiming', 'layer': 'ConvTranspose2d'}, {'type': 'Constant', 'layer': 'NaiveSyncBatchNorm2d', 'val': 1.0}]
2023-06-05 19:13:17,106 - mmdet - INFO - initialize Anchor3DHead with init_cfg {'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01, 'override': {'type': 'Normal', 'name': 'conv_cls', 'std': 0.01, 'bias_prob': 0.01}}
2023-06-05 19:13:17,107 - mmdet - INFO - Model:
VoxelNet(
  (backbone): SECOND(
    (blocks): ModuleList(
      (0): Sequential(
        (0): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (14): ReLU(inplace=True)
        (15): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (16): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (17): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (4): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (7): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (8): ReLU(inplace=True)
        (9): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (10): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (13): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (14): ReLU(inplace=True)
        (15): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (16): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (17): ReLU(inplace=True)
      )
    )
  )
  init_cfg={'type': 'Kaiming', 'layer': 'Conv2d'}
  (neck): SECONDFPN(
    (deblocks): ModuleList(
      (0): Sequential(
        (0): ConvTranspose2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
    )
  )
  init_cfg=[{'type': 'Kaiming', 'layer': 'ConvTranspose2d'}, {'type': 'Constant', 'layer': 'NaiveSyncBatchNorm2d', 'val': 1.0}]
  (bbox_head): Anchor3DHead(
    (loss_cls): FocalLoss()
    (loss_bbox): SmoothL1Loss()
    (loss_dir): CrossEntropyLoss(avg_non_ignore=False)
    (conv_cls): Conv2d(512, 18, kernel_size=(1, 1), stride=(1, 1))
    (conv_reg): Conv2d(512, 42, kernel_size=(1, 1), stride=(1, 1))
    (conv_dir_cls): Conv2d(512, 12, kernel_size=(1, 1), stride=(1, 1))
  )
  init_cfg={'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01, 'override': {'type': 'Normal', 'name': 'conv_cls', 'std': 0.01, 'bias_prob': 0.01}}
  (voxel_layer): Voxelization(voxel_size=[0.05, 0.05, 0.1], point_cloud_range=[0, -40, -3, 70.4, 40, 1], max_num_points=5, max_voxels=(16000, 40000), deterministic=True)
  (voxel_encoder): HardSimpleVFE()
  (middle_encoder): SparseEncoder(
    (conv_input): SparseSequential(
      (0): SubMConv3d()
      (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
    (encoder_layers): SparseSequential(
      (encoder_layer1): SparseSequential(
        (0): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer2): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(32, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer3): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
      (encoder_layer4): SparseSequential(
        (0): SparseSequential(
          (0): SparseConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (1): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
        (2): SparseSequential(
          (0): SubMConv3d()
          (1): BatchNorm1d(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
        )
      )
    )
    (conv_out): SparseSequential(
      (0): SparseConv3d()
      (1): BatchNorm1d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
)
```   
## 0606:  
1. [KITTI 3D 目标检测排行榜](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)  
## 0610：  
1. 最近在复现VirConv模型，复现的时候发现没有`semi.txt`，于是自己写了一个脚本，生成了该文件（查看论文发现它使用了10888，即semi中所有的文件来训练，因此直接把文件的name都写入txt即可）。  
```
import os

file_path = "/media/xilm/Lenovo/data/kitti/semi/image_2"
save_path = "/media/xilm/Lenovo/data/kitti/ImageSets/semi.txt"
file_names = os.listdir(file_path)
f = open(save_path,"a")
for file_name in file_names:
    name = file_name.split(".")[0]
    f.write(name)
    f.write("\r\n")
```  
- 此时执行`python3 -m pcdet.datasets.kitti.kitti_datasetsemi create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml`就可以了。  
## 0611：  
1. 实在是解决不了train.py运行时的这个问题了：  
```
Traceback (most recent call last):                   | 7/1856 [00:17<54:19,  1.76s/it, total_it=7]
  File "/home/xilm/fuxian/VirConv/tools/train.py", line 209, in <module>
    main()
  File "/home/xilm/fuxian/VirConv/tools/train.py", line 152, in main
    train_model(
  File "/home/xilm/fuxian/VirConv/tools/train_utils/train_utils.py", line 95, in train_model
    accumulated_iter = train_one_epoch(
  File "/home/xilm/fuxian/VirConv/tools/train_utils/train_utils.py", line 44, in train_one_epoch
    loss, tb_dict, disp_dict = model_func(model, batch)
  File "/home/xilm/fuxian/VirConv/pcdet/models/__init__.py", line 32, in model_func
    ret_dict, tb_dict, disp_dict = model(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/VirConv/pcdet/models/detectors/voxel_rcnn.py", line 10, in forward
    batch_dict = cur_module(batch_dict)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/VirConv/pcdet/models/backbones_3d/spconv_backbone.py", line 657, in forward
    newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)
  File "/home/xilm/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/home/xilm/fuxian/VirConv/pcdet/models/backbones_3d/spconv_backbone.py", line 215, in forward
    uv_coords, depth = index2uv(d3_feat2.indices, batch_size, calib, stride, x_trans_train, trans_param)
  File "/home/xilm/fuxian/VirConv/pcdet/models/backbones_3d/spconv_backbone.py", line 71, in index2uv
    pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts[:, 0:3])
  File "/home/xilm/fuxian/VirConv/pcdet/utils/calibration_kitti.py", line 128, in lidar_to_rect_cuda
    pts_rect = torch.matmul(pts_lidar_hom, torch.matmul(V2C, R0))
RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`
```  
## 0619:  
1. [PointRCNN讲解](https://blog.csdn.net/wqwqqwqw1231/article/details/90788500?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-6.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~BlogCommendFromBaidu~default-6.no_search_link&utm_relevant_index=10)
2. [PointRCNN讲解2](https://blog.csdn.net/qingliange/article/details/122379777)  
