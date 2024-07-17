import argparse
import shutil
from os import listdir
from os.path import abspath, basename, dirname, isfile, join, splitext

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


def process(file: str, result_folder: str):
    project_path = dirname(abspath(join(__file__, "..")))

    split_outpath = join(project_path, "out")

    demucs.separate.main(["-n", MODEL, file, "-o", split_outpath])

    input_file_name, input_ext = splitext(basename(file))

    outpath_actual = join(split_outpath, MODEL, input_file_name)

    print("loading drums..")
    sr, drums = wavefile.load(join(outpath_actual, "drums.wav"))
    print("loading bass..")
    sr, bass = wavefile.load(join(outpath_actual, "bass.wav"))
    print("loading other..")
    sr, other = wavefile.load(join(outpath_actual, "other.wav"))
    print("loading vocals..")
    sr, vocals = wavefile.load(join(outpath_actual, "vocals.wav"))

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
    write(join(result_folder, input_file_name + " (drumless).mp3"), sr, drumless)

    print("saving extra drums..")
    write(join(result_folder, input_file_name +
          " (extra drums).mp3"), sr, extra_drum)

    print("saving full..")
    shutil.copy(file, join(result_folder,
                input_file_name + " (full)" + input_ext))


def main():
    parser = argparse.ArgumentParser(description='drumscript')
    parser.add_argument('-i', help='input file/folder', required=True)
    parser.add_argument('-o', help='output folder', required=True)
    args = vars(parser.parse_args())

    print(f"Processing from: '{args['i']}'")
    print(f"Storing to: '{args['o']}'")

    files = []
    if isfile(args["i"]):
        files.append(abspath(args["i"]))
    else:
        for file in listdir(args["i"]):
            file = join(args["i"], file)
            if isfile(file) and not file in [".", ".."]:
                files.append(file)


    print(f"Processing {len(files)} files:")
    for file in files:
        print(f"  {file}")
    
    for file in files:
        process(file, args["o"])


if __name__ == '__main__':
    main()
