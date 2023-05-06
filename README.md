# pointcloud_paper
Some papers about pointcloud 3D detection  
## 0505:  
1. VoxelNet:[Pytorch实现](https://github.com/skyhehe123/VoxelNet-pytorch)  
- 学习了点云与图像如何进行对齐，并使用Matlab code进行了数据校准，将该代码使用Python进行实现。  
- Matlab绘制结果  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/figure/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230505203558.png)  
- Python绘制结果  
![](https://github.com/XxxuLimei/pointcloud_paper/blob/main/caliberation_0505/Python/tmp2avsrejr.PNG)  
## 0506:  
1. 发现上面那个仓库的代码在编译nms model的时候失败了，于是换了一个[仓库](https://github.com/RPFey/voxelnet_pytorch)  
- 首先Pip detectron2;  
