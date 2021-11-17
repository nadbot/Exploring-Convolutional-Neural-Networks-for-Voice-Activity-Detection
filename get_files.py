import glob
import itertools
from typing import List

from Enums import AudioClipFileLength, RecPlace, SNR


def get_available_files(path_base: str, snr: SNR = SNR.ZERO,
                        audio_clip_length_files: AudioClipFileLength = AudioClipFileLength.ONE_MINUTE,
                        rec_place: RecPlace = RecPlace.a) -> List[str]:
    """
    Function that finds the files in a given path for the given snr, file length and recording place.
    Args:
        path_base: Path to the base directory of the data
        snr: Enum describing the level of noise that exists in the data
        audio_clip_length_files: Enum describing the length of the audio clips
        rec_place: Enum describing the place of recording, either A or B.
    Returns:
        wav_files: List of filenames
    """
    locations = glob.glob(path_base + "*/")
    wav_files = []
    for folder in locations:
        path = folder + rec_place.value + str(audio_clip_length_files.value) + snr.value
        wavs = glob.glob(path + "/*.wav")
        if len(wavs) > 1:
            wav_files.append(wavs)
    wav_files = list(itertools.chain.from_iterable(wav_files))

    print(f'Total length: {len(wav_files)}')
    return wav_files
