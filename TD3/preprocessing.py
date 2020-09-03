import os
import os.path
import librosa
import math
import json

TIMIT_PATH = "/Users/soankim/Downloads/TIMIT/data/TRAIN/"
NOISE_PATH = '/Users/soankim/Downloads/ESC-50(2000)/audio(2000)/'
JSON_PATH = '/Users/soankim/PycharmProjects/Actor-Critic-Methods-Paper-To-Code/TD3/data.json'
SAMPLE_RATE = 22050
DURATION = 1  # measured in second
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(path_name, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # if 1.2 --> 2.0
    # dictionary to store data
    data = {"mfcc": [],
            "labels": []}

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(path_name)):
        for filename in [f for f in filenames if f.endswith("WAV.wav")]:
            if i <= 2:
                file_dirs = os.path.join(dirpath, filename)
                signal, sr = librosa.load(file_dirs, sr=SAMPLE_RATE)
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s  # s = 0 --> 0
                    finish_sample = start_sample + num_samples_per_segment  # s = 0 -> num_samples per segments

                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], sr=SAMPLE_RATE,
                                                n_fft=n_fft,
                                                n_mfcc=n_mfcc,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(path_name[25:30])
                        #print("{}, segment:{}".format(path_name, str(s+1)))

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(TIMIT_PATH, JSON_PATH, num_segments=10)
