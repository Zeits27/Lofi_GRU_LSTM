import numpy as np
import soundfile as sf

sr = 44100
duration = 8.0  # seconds
samples = int(sr * duration)

# base noise
noise = np.random.randn(samples) * 0.002

# random crackles
for _ in range(80):
    pos = np.random.randint(0, samples - 200)
    length = np.random.randint(20, 120)
    noise[pos:pos+length] += np.random.randn(length) * 0.05

# low-pass vinyl tone
from scipy.signal import butter, lfilter
b, a = butter(2, 6000 / (sr / 2), btype="low")
noise = lfilter(b, a, noise)

noise = np.clip(noise, -1.0, 1.0)

sf.write("vinyl_crackle_loop.wav", noise, sr)
print("✅ vinyl_crackle_loop.wav")
