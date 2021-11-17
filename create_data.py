import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow.keras.utils import to_categorical

from Enums import SNR, AudioClipFileLength, RecPlace
from get_files import get_available_files
from get_rgb_spectrograms import get_spectrograms_all_files


def reshape_specs(specs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data and labels to the correct shapes:
    Args:
        specs: Array of shape (len(wav_files), frames_per_file, 128, 32)
                where the last two are the size of the spectrogram
        labels: Array of shape (len(wav_files), frames_per_file)
                with elements being either 0 for no speech or 1 for speech
    Returns:
        specs: Array where the first two dimensions are merged
        labels: Array where the first two dimensions are merged and the values are one-hot encoded

    """
    specs_new = specs.reshape(specs.shape[0] * specs.shape[1], specs.shape[2], specs.shape[3])
    labels = labels.reshape(labels.shape[0] * labels.shape[1])
    labels = to_categorical(labels)
    return specs_new, labels


def create_data(path_data: str, path_save: str, snr: SNR = SNR.ZERO, recreate: bool = False) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create the data from running get_rgb_spectrograms. If data exists and recreate is False, load the data instead.
    Args:
        path_data: Path to the directory where the QUT-NOISE-TIMIT data is stored
        path_save: Path to which the data as well as the rgb_spectrograms should be saved
        snr: Enum describing the level of noise that exists in the data
        recreate: If true, ignore the saved files and recreate the rgb_spectrogram data from scratch
    Returns:
        x_train: np array containing the rgb spectrograms for each frame from the training dataset
        y_train: np array containing the label for each frame of the training dataset
        x_test: np array containing the rgb spectrograms for each frame from the test dataset
        y_test: np array containing the label for each frame of the test dataset
        frames_all: np array containing the raw audio for each frame from the training dataset
        frames_all_test: np array containing the raw audio for each frame from the test dataset

    """
    saved_files = [
        f'{path_save}x_train_{snr.value}.p',
        f'{path_save}y_train_{snr.value}.p',
        f'{path_save}x_test_{snr.value}.p',
        f'{path_save}y_test_{snr.value}.p',
        f'{path_save}frames_{snr.value}.p',
        f'{path_save}frames_test_{snr.value}.p',
    ]

    # check if all data exists. If one doesn't, recreate the data from scratch
    paths = [Path(filename) for filename in saved_files]
    files_exist = [path.is_file() for path in paths]
    if not all(files_exist) or recreate:
        # get the filenames of the training and testing datasets
        wav_files_train = get_available_files(path_data, snr, AudioClipFileLength.ONE_MINUTE,
                                              rec_place=RecPlace.a)
        wav_files_test = get_available_files(path_data, snr, AudioClipFileLength.ONE_MINUTE,
                                             rec_place=RecPlace.b)

        # extract the spectrograms as well as labels and raw audio frames
        specs_all, specs_np_all, labels_all, frames_all = get_spectrograms_all_files(wav_files_train[0:3],
                                                                                     path_save, snr)
        specs_all_test, specs_np_all_test, labels_all_test, frames_all_test = get_spectrograms_all_files(
            wav_files_test[0:2], path_save, snr)

        # reshape the data to fit the model input
        x_train, y_train = reshape_specs(specs_np_all, labels_all)
        x_test, y_test = reshape_specs(specs_np_all_test, labels_all_test)

        # as the elements are recreated, save them all and overwrite existing dataframes.
        pickle.dump(x_train, open(saved_files[0], "wb"))
        pickle.dump(y_train, open(saved_files[1], "wb"))
        pickle.dump(x_test, open(saved_files[2], "wb"))
        pickle.dump(y_test, open(saved_files[3], "wb"))
        pickle.dump(frames_all, open(saved_files[4], "wb"))
        pickle.dump(frames_all_test, open(saved_files[5], "wb"))

    else:
        # if data already exists and should not be overwritten, load the existing data
        x_train = pickle.load(open(saved_files[0], "rb"))
        y_train = pickle.load(open(saved_files[1], "rb"))
        x_test = pickle.load(open(saved_files[2], "rb"))
        y_test = pickle.load(open(saved_files[3], "rb"))
        frames_all = pickle.load(open(saved_files[4], "rb"))
        frames_all_test = pickle.load(open(saved_files[5], "rb"))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test, frames_all, frames_all_test
