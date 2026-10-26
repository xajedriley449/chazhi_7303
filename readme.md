原培轩的理解
data_show_energy trans.py
input：7303_xyzet.csv
output：7303_xyzet_rotated.csv
作用是将原始点云进行变换以对齐工作面

script_find_t_points:
流程：find_dense_projections.py -> dbscan_from_dense.py -> ransac_from_dbscan.py

1. find_dense_projections.py
input：data/7303_xyzet_rotated.csv
output：script_find_t_points/result/density_classification_results.csv
作用是用三维核密度估计+阈值分割得到的高密度和低密度时空点云的划分，

2. dbscan_from_dense.py
input：script_find_t_points/result/density_classification_results.csv
output：script_find_t_points/result/dbscan_clustering_results.csv
作用是对高密度时空点云用dbscan得到的dbscan点云聚类簇，

3. ransac_from_dbscan.py
input：script_find_t_points/result/dbscan_clustering_results.csv
output：script_find_t_points/result/ransac_results.csv script_find_t_points/result/line_parameters.csv
作用是对dbscan点云聚类簇用ransac直线拟合得到时序轨迹线点云簇，

4. t_calculate_distance.py
input: data/7303_xyzet_rotated.csv script_find_t_points/result/line_parameters.csv
output: script_calculate_distance/result/t_point_distances.csv
作用是计算各微震点到时间轨迹线的最短距离


script_find_xyz_points:
流程：denoise_point_cloud.py -> fit_local_planes.py -> cluster_planes.py
1. denoise_point_cloud.py
input：data/7303_xyzet_rotated.csv
output：script_find_xyz_points/result/01_denoised_points.csv
作用是使用DBSCAN去噪（标记噪声点，不删除）

2. fit_local_planes.py
input：script_find_xyz_points/result/01_denoised_points.csv
output：script_find_xyz_points/result/02_local_planes.csv
作用是对保留点拟合局部平面（基于去噪结果），即计算各点法向量，计算非噪声点到局部平面的距离

3. cluster_planes.py
input：script_find_xyz_points/result/02_local_planes.csv
output：script_find_xyz_points/result/03_segmented_planes.csv
作用是聚类合并局部平面，得到所谓分割平面，计算所有点（包括噪声点）到分割平面的距离

4. xyz_calculate_distance.py
input: script_find_xyz_points/result/03_segmented_planes.csv
output: script_calculate_distance/result/xyz_point_distances.csv
作用是计算各微震点到空间聚类面的最短距离

更高层次的流程:
script_find_t_points / script_find_xyz_points 提取时间轨迹线/空间聚类面等协变量特征
-> chazhi.py / chazhi_cov.py / chazhi_cov2.py 整合协变量特征与微震按日切片插值为格网图像，并对图像插帧
-> pred_tx.py / pred_tx_cov.py / pred_tx_cov2.py 预测+线=cov，预测+面=cov2，tx/ty/tz
-> pred_tx_cov3.py / pred_tx_cov3_inter.py 预测+线+面=cov3(前期融合)/cov3_inter(中期融合)