import rosbag
import sys
import argparse

# --- SCRIPT CONFIGURATION ---
# These values are from your frames.pdf
#
# The "world" or "map" frame. The fixed origin.
TARGET_FRAME = "world" 
# The frame of the drone's body/IMU whose pose you want to track.
SOURCE_FRAME = "imu_link" 
# The topic where the transform information is published.
TF_TOPIC = "/tf"
# -----------------------------


def main():
    """
    Extracts pose information from a ROS bag file and saves it in TUM format.
    Usage: python extract_poses.py your_bag.bag output_poses.txt
    """
    parser = argparse.ArgumentParser(description="Extract pose data from a ROS bag file into TUM format.")
    parser.add_argument("bag_file", help="Input ROS bag file.")
    parser.add_argument("output_file", help="Output text file for poses.")
    args = parser.parse_args()

    print(f"Reading bag file: {args.bag_file}")
    print(f"Looking for transform from '{TARGET_FRAME}' (world) to '{SOURCE_FRAME}' (body)")

    try:
        bag = rosbag.Bag(args.bag_file, "r")
    except rosbag.bag.ROSBagUnindexedException:
        print("\nERROR: Bag file is unindexed. Please run 'rosbag reindex your_bag_file.bag' first.")
        return
        
    count = 0
    
    with open(args.output_file, 'w') as out_f:
        # Iterate through all messages in the bag
        for topic, msg, t in bag.read_messages(topics=[TF_TOPIC]):
            # The /tf topic contains a list of transforms
            for transform in msg.transforms:
                # Check if this is the transform we are looking for
                if transform.header.frame_id == TARGET_FRAME and transform.child_frame_id == SOURCE_FRAME:
                    
                    # Timestamp
                    secs = transform.header.stamp.secs
                    nsecs = transform.header.stamp.nsecs
                    timestamp = f"{secs}.{nsecs:09d}"

                    # Translation (position)
                    tx = transform.transform.translation.x
                    ty = transform.transform.translation.y
                    tz = transform.transform.translation.z

                    # Rotation (as a quaternion)
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    # Write in TUM format: timestamp tx ty tz qx qy qz qw
                    out_f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
                    count += 1

    bag.close()
    
    if count == 0:
        print("\n--- WARNING ---")
        print("Extraction complete, but 0 poses were found.")
        print("Please double-check your TARGET_FRAME and SOURCE_FRAME names in the script.")
        print(f"Expected: TARGET='{TARGET_FRAME}', SOURCE='{SOURCE_FRAME}'")
        print("Make sure they exactly match your frames.pdf (no extra slashes '/')")
    else:
        print(f"\nDone. Extracted {count} poses to {args.output_file}")


if __name__ == '__main__':
    main()