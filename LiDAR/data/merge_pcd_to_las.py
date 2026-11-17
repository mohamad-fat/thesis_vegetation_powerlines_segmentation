import open3d as o3d
import numpy as np
import laspy
import glob

pcd_folder = "/data/pcd_out"
output_las = "/data/voo_3_electrical_merged.las"

pcd_files = sorted(glob.glob(f"{pcd_folder}/*.pcd"))

all_points = []

for f in pcd_files:
    pcd = o3d.io.read_point_cloud(f)
    points = np.asarray(pcd.points)
    all_points.append(points)

all_points = np.vstack(all_points)

header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)
las.x = all_points[:,0]
las.y = all_points[:,1]
las.z = all_points[:,2]

las.write(output_las)
print(f"Merged {len(pcd_files)} PCD files into {output_las} with {all_points.shape[0]} points.")