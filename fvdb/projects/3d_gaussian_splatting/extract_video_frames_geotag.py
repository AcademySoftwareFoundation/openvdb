# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import os
import shutil

import cv2
import srt
import tyro
from tqdm import tqdm


def get_meta_data(meta_file):
    """grabs the Latitude, longitude and altitude from the SRT file and returns a list of them

    Args:
        meta_file (str): SRT file with info as such
        "[latitude: 42.558562] [longitude: -2.163064] [rel_alt: 28.400 abs_alt: 404.972]"
        some where in the file for each image

    Returns:
         3 Lists, 1 for each Latitude, longitude and altitude, elements for each image
    """
    with open(meta_file, "r", encoding="utf-8") as f:
        srt_data = f.read()

    meta_data_list = list(srt.parse(srt_data))
    print((meta_data_list[0]))
    lats = []
    longs = []
    alts = []
    lat_str = "latitude: "
    long_str = "longitude: "
    alt_str = "abs_alt: "
    for i in range(0, len(meta_data_list)):
        image_meta_data_str = str(meta_data_list[i])
        # print(image_meta_data_str)

        # Latitude
        lat_index = image_meta_data_str.find(lat_str) + len(lat_str)
        temp_str = image_meta_data_str[lat_index:-1]
        split_str = temp_str.split("]")
        lats.append(float(split_str[0]))
        # print(lats[i])

        # Longitude
        long_index = image_meta_data_str.find(long_str) + len(long_str)
        temp_str = image_meta_data_str[long_index:-1]
        split_str = temp_str.split("]")
        longs.append(float(split_str[0]))

        # altitude
        alt_index = image_meta_data_str.find(alt_str) + len(alt_str)
        alt_str = image_meta_data_str[alt_index:-1]
        split_str = temp_str.split("]")
        alts.append(float(split_str[0]))
        # print(longs[i])
    return lats, longs, alts


def save_frames(input_path, exiftool_path, lats, longs, alts, output_dir="output", frame_freq=100):
    """saves frames and their lat, long, alt meta data

    Args:
        input_path (str): path to video file
        exiftool_path (str): path to exiftool binary
        lats (list): list of floats of latitude values for each image
        longs (_type_): list of floats of longitude values for each image
        alts (_type_): list of floats of altitude values for each image
        output_dir (str, optional): Directory to save images to. Will create if DNE. Defaults to "output".
        frame_freq (int, optional): How frequently to write out frames. Defaults to 100.
    """

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    video_capture = cv2.VideoCapture(input_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # if len(lats) != frame_count or len(longs) != frame_count or len(alts) != frame_count:
    #     raise ValueError("GPS info does not match video length")

    frame_count = 0

    pbar = tqdm(desc="Extracting frames", total=frame_count)
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        if frame_count % frame_freq == 0:
            collect_name = os.path.basename(output_dir)
            frame_path = os.path.join(output_dir, f"{frame_count:06d}_{collect_name}.jpg")
            cv2.imwrite(frame_path, frame)
            lat_tag = "GPSLatitude"
            lon_tag = "GPSLongitude"
            alt_tag = "GPSAltitude"

            latVal = lats[frame_count]
            lonVal = longs[frame_count]
            altVal = alts[frame_count]

            if latVal != 0 and lonVal != 0:

                cmd = f"{exiftool_path} {frame_path} -overwrite_original -{lat_tag}={latVal} -{lon_tag}={lonVal} -{alt_tag}={altVal} -GPSLongitudeRef=W -GPSAltitudeRef=0"
                print(cmd)

                exit_code = os.system(cmd)

                if exit_code != 0:
                    print(f"exiftool failed with code {exit_code}. Exiting.")
                    exit(exit_code)
                print(
                    "Exif data modified for frame: ",
                    frame_path,
                    "with GPS Loc: (" + str(latVal) + ", " + str(lonVal) + ")",
                )

        frame_count += 1
        pbar.update(1)

    video_capture.release()
    print(f"Extracted {frame_count} frames to {output_dir}")


def main(srt_paths: list[str], vid_paths: list[str], frame_freq: int):
    """
    Script to geotag frames from a video with locations from corresponding SRT file
    Args:
        srt_paths: list of paths to SRT Files
        vid_paths: list of paths to video Files
        frame_freq: how often to write a geotagged frame to disk from video
    """
    for srt_path, vid_path in zip(srt_paths, vid_paths):
        output_path = os.path.splitext(vid_path)[0]
        lats, longs, alts = get_meta_data(srt_path)
        save_frames(vid_path, lats, longs, alts, output_path, frame_freq=frame_freq)


if __name__ == "__main__":
    tyro.cli(main)
