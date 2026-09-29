#!/usr/bin/env python3
"""Build the files that go on a TV USB stick.

TVs do not run programs from USB; they play media files with their built-in
player. So the "program" is a file layout every TV media player understands:

  - H.264 High / yuv420p MP4 with a silent AAC stereo track (some TV players
    refuse or stall on video-only files),
  - the seamless 30 s clip repeated end-to-end into one long file, so the
    display keeps looping even on TVs without a Repeat setting,
  - short 8.3 style upper-case file names in the drive root.

Outputs (in ./usb/):
  HYPERVAN_4K.mp4     3840x2160, level 5.1, 4 minutes
  HYPERVAN_HD.mp4     1920x1080, level 4.1, 14 minutes
"""
import os
import subprocess
import tempfile

import imageio_ffmpeg

HERE = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(HERE, "hypervan_hbot_sequence_4k.mp4")
OUT = os.path.join(HERE, "usb")
FF = imageio_ffmpeg.get_ffmpeg_exe()


def run(*args):
    subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True)


def loop(src, repeats, dst, tmp):
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        f.write(f"file '{src}'\n" * repeats)
    run("-f", "concat", "-safe", "0", "-i", lst,
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "32k",
        "-shortest", "-movflags", "+faststart", dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        hd = os.path.join(tmp, "hd.mp4")
        run("-i", CLIP, "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264",
            "-preset", "slow", "-crf", "18", "-profile:v", "high", "-level", "4.1",
            "-pix_fmt", "yuv420p", "-g", "300", hd)
        loop(CLIP, 8, os.path.join(OUT, "HYPERVAN_4K.mp4"), tmp)
        loop(hd, 28, os.path.join(OUT, "HYPERVAN_HD.mp4"), tmp)
    for name in sorted(os.listdir(OUT)):
        print(name, os.path.getsize(os.path.join(OUT, name)))


if __name__ == "__main__":
    main()
