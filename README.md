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
