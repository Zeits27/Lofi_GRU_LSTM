import subprocess


midi_file = "lofi_output_balanced.mid"
soundfont = "sf/piano.sf2"
output = "test3.wav"


cmd = [
    "fluidsynth",
    "-ni",
    "-F", output,
    "-r", "44100",
    "-g", "0.5",    
    "-C", "30",      
    "-R", "40",       
    soundfont,
    midi_file,
]

print("> Rendering MIDI with FluidSynth...")
subprocess.run(cmd, check=True)
print("> Done!")
print(f"> Wrote file: {output}")
