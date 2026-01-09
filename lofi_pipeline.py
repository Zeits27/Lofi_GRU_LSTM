import pickle
import subprocess
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from pedalboard import Pedalboard, LowpassFilter, Compressor, Distortion
from pedalboard.io import AudioFile
import os

from lofi_music_generator import (
    create_sequences, 
    create_network, 
    generate_notes, 
    create_midi
)


class LofiPipeline:
    def __init__(self, 
                 weights_file='full_model.hdf5',
                 soundfont='sf/piano.sf2',
                 vinyl_sfx='./sfx/vinyl_crackle_loop.wav',
                 drums_sfx='./sfx/drums.wav',
                 rain_sfx='./sfx/rain1.wav'):
        
        self.weights_file = weights_file
        self.soundfont = soundfont
        self.vinyl_sfx = vinyl_sfx
        self.drums_sfx = drums_sfx
        self.rain_sfx = rain_sfx
        
    
    def generate_midi(self, length=200, top_k=6, temperature=0.85):
        print("\n[1/5] Generating MIDI...")
        
        # Load data
        with open('data/notes.pkl', 'rb') as f:
            notes = pickle.load(f)
        
        with open('data/mappings.pkl', 'rb') as f:
            mappings = pickle.load(f)
            int_to_note = mappings['int_to_note']
            n_vocab = mappings['n_vocab']
        
        network_input, _, _, _ = create_sequences(notes, sequence_length=64)
        
        print("Creating model architecture...")
        model = create_network(network_input, n_vocab)
        
        print(f"Loading weights from {self.weights_file}...")
        model.load_weights(self.weights_file, skip_mismatch=True)
        print("✓ Model loaded successfully!")
        
        print(f"Generating {length} notes...")
        generated_notes = generate_notes(
            model, network_input, int_to_note, n_vocab,
            length=length, top_k=top_k, temperature=temperature
        )
        
        midi_file = "temp_lofi.mid"
        create_midi(generated_notes, filename=midi_file)
        print(f"✓ MIDI saved: {midi_file}")
        return midi_file
    
    
    def midi_to_wav(self, midi_file):
        print("\n[2/5] Rendering MIDI to WAV...")
        
        output = "temp_raw.wav"
        
        cmd = [
            "fluidsynth",
            "-ni",
            "-F", output,
            "-r", "44100",
            "-g", "0.5",
            "-C", "30",
            "-R", "40",
            self.soundfont,
            midi_file,
        ]
        
        subprocess.run(cmd, check=True)
        print(f"✓ WAV rendered: {output}")
        return output
    
    
    def soften_audio(self, input_wav):
        print("\n[3/5] Softening audio...")
        
        audio = AudioSegment.from_wav(input_wav)
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
        
        output = "temp_soft.wav"
        soft.export(output, format="wav")
        print(f"✓ Softened: {output}")
        return output
    
    
    def apply_lofi_fx(self, input_wav):
        print("\n[4/5] Applying lofi effects...")
        
        audio, sr = sf.read(input_wav, always_2d=True)
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
        
        output = "temp_lofi.wav"
        with AudioFile(output, "w", samplerate=sr, num_channels=lofi.shape[0], bit_depth=16) as f:
            f.write(lofi)
        
        print(f"✓ Lofi FX applied: {output}")
        return output
    
    
    def mix_final(self, music_wav, output_file="FINAL_LOFI.wav"):
        print("\n[5/5] Mixing final track...")
        
        music, sr = sf.read(music_wav, always_2d=True)
        music = music.T.astype(np.float32)
        
        # Vinyl crackle
        vinyl, _ = sf.read(self.vinyl_sfx, always_2d=True)
        vinyl = vinyl.T.astype(np.float32)
        if vinyl.shape[0] == 1:
            vinyl = np.vstack([vinyl, vinyl])
        repeat = int(np.ceil(music.shape[1] / vinyl.shape[1]))
        vinyl = np.tile(vinyl, (1, repeat))[:, :music.shape[1]]
        
        music_with_vinyl = np.clip(music + vinyl * 0.12, -1.0, 1.0)
        
        # Drums
        drums, _ = sf.read(self.drums_sfx, always_2d=True)
        drums = drums.T.astype(np.float32)
        repeat = int(np.ceil(music_with_vinyl.shape[1] / drums.shape[1]))
        drums = np.tile(drums, (1, repeat))[:, :music_with_vinyl.shape[1]]
        
        # Rain
        rain, _ = sf.read(self.rain_sfx, always_2d=True)
        rain = rain.T.astype(np.float32)
        if rain.shape[0] == 1:
            rain = np.vstack([rain, rain])
        repeat = int(np.ceil(music_with_vinyl.shape[1] / rain.shape[1]))
        rain = np.tile(rain, (1, repeat))[:, :music_with_vinyl.shape[1]]
        
        # Final mix
        final = (
            music_with_vinyl * 0.9
            + drums * 0.6
            + rain * 0.03
        )
        final = np.clip(final, -1.0, 1.0)
        
        sf.write(output_file, final.T, sr)
        print(f"Final track saved: {output_file}")
        return output_file
    
    
    def cleanup_temp_files(self):
        temp_files = ["temp_lofi.mid", "temp_raw.wav", "temp_soft.wav", "temp_lofi.wav"]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
                print(f"Cleaned: {f}")