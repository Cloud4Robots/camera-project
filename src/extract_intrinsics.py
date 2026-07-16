import glob
import os
import pyrealsense2 as rs
# read through all .bag files in the data folder and print out the intrinsics for each stream

def print_intrinsics(bag_path: str) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    # take the first frame to get the intrinsics, then stop the pipeline
    print(f"\n=== {bag_path} ===")
    try:
        profile = pipeline.start(config)

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"depth_scale: {depth_scale}")

        for stream in profile.get_streams():
            intr = stream.as_video_stream_profile().get_intrinsics()
            print(f"{stream.stream_name():6} {intr.width}x{intr.height}  "
                  f"fx={intr.fx:.2f} fy={intr.fy:.2f} "
                  f"ppx={intr.ppx:.2f} ppy={intr.ppy:.2f}  "
                  f"model={intr.model} coeffs={intr.coeffs}")

        extr = profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
        print(f"depth->color  rotation={extr.rotation}  translation={extr.translation}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    bag_files = sorted(glob.glob(os.path.join("../data", "*.bag")))
    if not bag_files:
        raise FileNotFoundError("No .bag file found in ../data")

    print(f"Found {len(bag_files)} .bag file(s)")
    for bag_path in bag_files:
        print_intrinsics(bag_path)