import argparse
import json

from mcap_ros2.reader import read_ros2_messages


TOPIC = "/camera/color/camera_info"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    for msg in read_ros2_messages(args.bag, topics=[TOPIC]):
        camera_info = msg.ros_msg

        result = {
            "cam_K": [
                float(x) for x in camera_info.k
            ],
            "depth_scale": 1.0,
        }

        with open(args.output, "w") as f:
            json.dump({"000000": result}, f, indent=2)

        print("Saved camera calibration:")
        print(json.dumps(result, indent=2))
        return

    raise RuntimeError(f"No messages found on {TOPIC}")


if __name__ == "__main__":
    main()