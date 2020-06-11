from motion_visualizer.read_bvh import read_bvh_to_array
import numpy as np
import argparse


def convert_bvh2npy(bvh_file, npy_file):
    # read the coordinates
    coords = read_bvh_to_array(bvh_file)

    # shorten the sequence
    output_vectors = coords[:6000]

    np.save(npy_file, output_vectors)


if __name__ == "__main__":

    # Parse command line params
    parser = argparse.ArgumentParser(
        description="Transforming BVH file into an array of feature-vectors"
    )

    # Folders params
    parser.add_argument(
        "--bvh_file",
        "-bvh",
        default="/home/taras/Documents/Datasets/SpeechToMotion/Irish/raw/Cleaned&Fixed/BVH_60fps/_1/NaturalTalking_001.bvh",
        help="Address to the bvh file",
    )
    parser.add_argument(
        "--npy_file", "-npy", default="ges_test.npy", help="Address to the npy file"
    )

    args = parser.parse_args()

    convert_bvh2npy(args.bvh_file, args.npy_file)
