import soundfile as sf
import numpy as np


music, sr = sf.read("soft_lofi_slow_pedal.wav", always_2d=True)
music = music.T.astype(np.float32)   # (channels, samples)


vinyl, _ = sf.read("./sfx/vinyl_crackle_loop.wav", always_2d=True)
vinyl = vinyl.T.astype(np.float32)

if vinyl.shape[0] == 1:
    vinyl = np.vstack([vinyl, vinyl])

repeat = int(np.ceil(music.shape[1] / vinyl.shape[1]))
vinyl = np.tile(vinyl, (1, repeat))[:, :music.shape[1]]

vinyl_gain = 0.12
music_with_vinyl = np.clip(music + vinyl * vinyl_gain, -1.0, 1.0)


drums, _ = sf.read("./sfx/drums.wav", always_2d=True)
drums = drums.T.astype(np.float32)

repeat = int(np.ceil(music_with_vinyl.shape[1] / drums.shape[1]))
drums = np.tile(drums, (1, repeat))[:, :music_with_vinyl.shape[1]]


rain, _ = sf.read("./sfx/rain1.wav", always_2d=True)
rain = rain.T.astype(np.float32)

if rain.shape[0] == 1:
    rain = np.vstack([rain, rain])

repeat = int(np.ceil(music_with_vinyl.shape[1] / rain.shape[1]))
rain = np.tile(rain, (1, repeat))[:, :music_with_vinyl.shape[1]]


final = (
    music_with_vinyl * 0.9
    + drums * 0.6
    + rain * 0.03        
)

final = np.clip(final, -1.0, 1.0)

sf.write("FINAL_LOFI.wav", final.T, sr)
print("✅ FINAL_LOFI2.wav created")
