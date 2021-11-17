from enum import Enum


class AudioClipFileLength(Enum):
    ONE_MINUTE = '/l060/'
    TWO_MINUTES = '/l120/'


class SNR(Enum):
    MINUS_TEN = 'n-10'
    MINUS_FIVE = 'n-05'
    ZERO = 'n+00'
    FIVE = 'n+05'
    TEN = 'n+10'
    FIFTEEN = 'n+15'


class RecPlace(Enum):
    a = 'sA'
    b = 'sB'
