import subprocess

cmd = [
    "fluidsynth",
    "-ni",
    "-F", "drums.wav",
    "-r", "44100",
    "-g", "0.7",
    "sf/FluidR3_GM.sf2",
    "lofi_drums.mid",
]

subprocess.run(cmd, check=True)
print("✅ drums.wav created")
