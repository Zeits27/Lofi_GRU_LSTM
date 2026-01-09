import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, LowpassFilter, Compressor, Distortion
from pedalboard.io import AudioFile


audio, sr = sf.read("soft_slow_lofi2.wav", always_2d=True)
audio = audio.T.astype(np.float32)  



board = Pedalboard([
    LowpassFilter(4200),
    Distortion(drive_db=2.0),
    Compressor(
        threshold_db=-20,
        ratio=3,
        attack_ms=10,
        release_ms=120
    ),
])

lofi = board(audio, sr)
lofi = np.clip(lofi, -1.0, 1.0)


with AudioFile(
    "soft_lofi_slow_pedal.wav",
    "w",
    samplerate=sr,
    num_channels=lofi.shape[0],
    bit_depth=16
) as f:
    f.write(lofi)

