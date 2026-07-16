import glob
import os
import pyrealsense2 as rs
#test if the bag file can be opened and read correctly, 
# make sure the first 5 frames can be read without error

# Look for any .bag file inside the data folder (one level up from src/)
data_dir = "../data"
bag_files = glob.glob(os.path.join(data_dir, "*.bag"))

if not bag_files:
    raise FileNotFoundError(f"No .bag file found in {data_dir}")

# If there are multiple .bag files, just use the first one for now
bag_file_path = bag_files[0]
print(f"Found {len(bag_files)} .bag file(s) in {data_dir}: {bag_files}")
print(f"Using: {bag_file_path}")

pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file_path, repeat_playback=False)

profile = pipeline.start(config)
print("\nSuccessfully opened. Stream info:")
for stream in profile.get_streams():
    print(" -", stream)

for i in range(5):
    frames = pipeline.wait_for_frames()
    print(f"Frame {i+1}: {frames}")

pipeline.stop()