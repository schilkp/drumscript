import argparse
import os
import shutil

import demucs.separate
import numpy as np
import pydub
import wavefile

MODEL = "htdemucs_ft"
EXTRA_DRUMS_OTHERS_DB = -9
EXTRA_DRUMS_DB = 1


def write(filename, sr, x):
    x = np.transpose(x)
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1

    max_ampl = np.max(np.abs(x), axis=(0, 1))
    x = x/max_ampl

    y = np.int16(x * 2 ** 15)

    song = pydub.AudioSegment(
        y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(filename, format="mp3", bitrate="320k")


def main():
    parser = argparse.ArgumentParser(description='drumscript')
    parser.add_argument('-i', help='input file', required=True)
    parser.add_argument('-o', help='output folder', required=True)
    parser.add_argument('-t', help='title', required=True)
    args = vars(parser.parse_args())

    project_path = os.path.dirname(
        os.path.abspath(os.path.join(__file__, "..")))

    outpath = os.path.join(project_path, "out")

    demucs.separate.main(["-n", MODEL, args["i"], "-o", outpath])

    input_file_name, input_ext = os.path.splitext(os.path.basename(args["i"]))

    outpath_actual = os.path.join(outpath, MODEL, input_file_name)

    print("loading drums..")
    sr, drums = wavefile.load(os.path.join(outpath_actual, "drums.wav"))
    print("loading bass..")
    sr, bass = wavefile.load(os.path.join(outpath_actual, "bass.wav"))
    print("loading other..")
    sr, other = wavefile.load(os.path.join(outpath_actual, "other.wav"))
    print("loading vocals..")
    sr, vocals = wavefile.load(os.path.join(outpath_actual, "vocals.wav"))

    minlen = min([drums.shape[1], bass.shape[1],
                 other.shape[1], vocals.shape[1]])

    drums = drums[:, 0:minlen]
    bass = bass[:, 0:minlen]
    other = other[:, 0:minlen]
    vocals = vocals[:, 0:minlen]

    print("mixing..")

    drumless = bass + other + vocals
    extra_drum = (bass + other + vocals) * pow(10,
                                               EXTRA_DRUMS_OTHERS_DB/20) + drums * pow(10, EXTRA_DRUMS_DB/20)

    print("saving drumless..")
    write(os.path.join(args["o"], args["t"] + " (drumless).mp3"), sr, drumless)

    print("saving extra drums..")
    write(os.path.join(args["o"], args["t"] +
          " (extra drums).mp3"), sr, extra_drum)

    print("saving full..")
    shutil.copy(args["i"], os.path.join(
        args["o"], args["t"] + " (full)" + input_ext))

if __name__ == '__main__':
    main()
