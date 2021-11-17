from pathlib import Path
from typing import Tuple, List

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from Enums import SNR
from sound_viewer_tool import create_png, SpectrogramImage


def get_spectrograms(filename: str, save_path: str, snr: SNR = SNR.ZERO) -> \
        Tuple[List[SpectrogramImage], List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Get the spectrograms for each 80ms (a spectrogram shows 160ms of audio).
    A spectrogram has the size 32*128.
    Args:
        filename: Filename of the audio file
        save_path: Path to which the spectrograms should be saved
        snr: Enum describing the level of noise that exists in the data
    Returns:
        specs: List of Spectrogram Images
        specs_np: List of np arrays containing the values of the spectrograms
        frames_labels: np array containing the label of each frame
        frames: np array containing the raw audio of each frame
    """
    save_path = save_path + snr.value + '/' + filename.split('\\')[-1] + '/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    speech, sr = librosa.load(filename, sr=16000)

    eventlab = filename.replace(".wav", ".eventlab")
    df = pd.read_csv(eventlab, sep=" ", header=None)
    df = df.loc[df[2] == 'speech'].copy()
    target_sampling_rate = 16000
    df[0] *= target_sampling_rate
    df[1] *= target_sampling_rate
    label = np.array([0] * len(speech))
    for index, row in df.iterrows():
        label[round(row[0]):round(row[1])+1] = 1
    # 160ms: sr /1000 * ms (16000/1000*160)
    frame_size = int(16000/1000*160)  # 160ms
    overlap = int(frame_size/2)  # 50% overlap
    frames = []
    frames_labels = []
    i = 0
    specs = []
    specs_np = []
    index = 0
    image_width = 32
    image_height = 128
    fft_size = 2048
    f_max = 8000  # 22050,
    f_min = 10
    wavefile = 0
    palette = 1
    channel = 1
    window = "hanning"
    logspec = 0

    while (i+frame_size) < speech.shape[0]:
        frame = speech[i:i+frame_size]
        frames.append(frame)

        frame_label = label[i:i+frame_size]
        sums = np.sum(frame_label, axis=0)
        sums = sums > 1
        frame_label = sums.astype(int)
        frames_labels.append(frame_label)

        output_filename_w = save_path + str(index) + ".png"
        output_filename_s = save_path + str(index) + ".png"
        args = (frame, sr, output_filename_w, output_filename_s,
                image_width, image_height, fft_size,
                f_max, f_min, wavefile,
                palette, channel, window, logspec)
        t = create_png(*args)
        arr = np.array(t)
        specs.append(t)
        specs_np.append(arr)

        i = i + frame_size - overlap
        index += 1

    # Add last element, pad with 0s
    frame = [0] * frame_size
    frame[0:frame_size-overlap] = speech[i:i+frame_size-overlap]
    frames.append(frame)
    frames = np.array(frames)

    frame_label = [0] * frame_size
    frame_label[0:frame_size-overlap] = label[i:i+frame_size-overlap]
    sums = np.sum(frame_label, axis=0)
    sums = sums > 1
    frame_label = sums.astype(int)
    frames_labels.append(frame_label)
    frames_labels = np.array(frames_labels)

    frame = np.array(frame)
    output_filename_w = save_path + str(index) + ".png"
    output_filename_s = save_path + str(index) + ".png"
    args = (frame, sr, output_filename_w, output_filename_s,
            image_width, image_height, fft_size,
            f_max, f_min, wavefile,
            palette, channel, window, logspec)
    t = create_png(*args)
    specs.append(t)
    arr = np.array(t)
    specs_np.append(arr)

    return specs, specs_np, frames_labels, frames


def get_spectrograms_all_files(wav_files: List[str], save_path: str, snr: SNR = SNR.ZERO) -> \
        Tuple[List[List[SpectrogramImage]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Get spectrograms for a list of files
    Args:
        wav_files: List of filenames
        save_path: Directory to which output should be saved
        snr: Enum describing the level of noise that exists in the data
    Returns:
        specs_all: Nested list of Spectrogram Images
        specs_np_all: Np array containing the numpy representation of Spectrogram Images
        labels_all: Np array containing the labels for all frames
        frames_all: Np array containing all frames
    """
    specs_all, specs_np_all, labels_all, frames_all = [], [], [], []
    for filename in tqdm(wav_files):
        specs, specs_np, labels, frames = get_spectrograms(filename, save_path, snr)
        specs_all.append(specs)
        specs_np_all.append(specs_np)
        labels_all.append(labels)
        frames_all.append(frames)
    specs_np_all = np.array(specs_np_all)
    labels_all = np.array(labels_all)
    frames_all = np.array(frames_all)
    return specs_all, specs_np_all, labels_all, frames_all
