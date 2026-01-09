from pydub import AudioSegment, effects

audio = AudioSegment.from_wav("test3.wav")

audio = audio + 2

soft = effects.compress_dynamic_range(
    audio,
    threshold=-32.0,
    ratio=8.0,
    attack=50,
    release=500
)

soft = soft.high_pass_filter(80)
soft = soft.low_pass_filter(3600)

soft.export("soft_slow_lofi2.wav", format="wav")
