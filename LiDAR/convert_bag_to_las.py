from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import laspy

# === CONFIG ===
bag_path = '/data/voo_3_electrical.bag'   # your .bag file
topic = '/ouster/points'                  # the LiDAR topic inside the bag
output_las = '/data/voo_3_electrical.las' # output LAS file

points_all = []

with Reader(bag_path) as reader:
    for connection, timestamp, rawdata in reader.messages():
        if connection.topic == topic:
            msg = deserialize_cdr(rawdata, connection.msgtype)

            # Extract XYZ from PointCloud2 without sensor_msgs
            fields = {f.name: f.offset for f in msg.fields}
            point_step = msg.point_step
            data = msg.data  # raw bytes

            npoints = int(len(data) / point_step)
            xyz = np.empty((npoints, 3), dtype=np.float32)

            for i, field in enumerate(['x','y','z']):
                offset = fields[field]
                xyz[:, i] = np.frombuffer(
                    data[offset::point_step], dtype=np.float32, count=npoints
                )

            points_all.append(xyz)

points_all = np.vstack(points_all)

# Write to LAS
header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)
las.x, las.y, las.z = points_all[:,0], points_all[:,1], points_all[:,2]
las.write(output_las)

print(f"Saved {output_las} with {len(points_all)} points.")