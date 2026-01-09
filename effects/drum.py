import pretty_midi
import random

def create_lofi_drum_midi(filename="lofi_drums.mid", bpm=80, bars=16):
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    drums = pretty_midi.Instrument(program=0, is_drum=True)

    beat = 60 / bpm
    t = 0.0

    for bar in range(bars):
        # KICK - varied pattern
        drums.notes.append(pretty_midi.Note(velocity=random.randint(95, 105), pitch=36, start=t, end=t + beat * 0.3))
        drums.notes.append(pretty_midi.Note(velocity=random.randint(90, 100), pitch=36, start=t + beat * 2, end=t + beat * 2.3))
        
        # Extra kick every other bar
        if bar % 2 == 1:
            drums.notes.append(pretty_midi.Note(velocity=random.randint(70, 85), pitch=36, start=t + beat * 1.75, end=t + beat * 1.95))
            drums.notes.append(pretty_midi.Note(velocity=random.randint(75, 90), pitch=36, start=t + beat * 3.5, end=t + beat * 3.7))

        # SNARE - with ghost notes
        drums.notes.append(pretty_midi.Note(velocity=random.randint(85, 100), pitch=40, start=t + beat, end=t + beat * 1.2))
        drums.notes.append(pretty_midi.Note(velocity=random.randint(90, 105), pitch=40, start=t + beat * 3, end=t + beat * 3.2))
        
        # Ghost snares
        drums.notes.append(pretty_midi.Note(velocity=random.randint(30, 45), pitch=40, start=t + beat * 1.75, end=t + beat * 1.85))
        if bar % 4 != 0:
            drums.notes.append(pretty_midi.Note(velocity=random.randint(35, 50), pitch=37, start=t + beat * 2.5, end=t + beat * 2.6))

        # Hi-hats - varied velocities and swing
        for i in range(8):
            swing = random.uniform(0, 0.08) if i % 2 == 1 else 0
            vel = random.randint(50, 70) if i % 2 == 0 else random.randint(30, 50)
            drums.notes.append(pretty_midi.Note(velocity=vel, pitch=42, start=t + beat * 0.5 * i + swing, end=t + beat * 0.5 * i + 0.12))

        # Open hi-hat variations
        if bar % 2 == 0:
            drums.notes.append(pretty_midi.Note(velocity=random.randint(45, 60), pitch=46, start=t + beat * 1.5, end=t + beat * 2.2))
        else:
            drums.notes.append(pretty_midi.Note(velocity=random.randint(50, 65), pitch=46, start=t + beat * 2.5, end=t + beat * 3.2))

        # Ride/crash for flavor
        if bar % 8 == 0:
            drums.notes.append(pretty_midi.Note(velocity=random.randint(50, 70), pitch=49, start=t, end=t + beat * 4))
        
        # Percussion fills every 4 bars
        if bar % 4 == 3:
            drums.notes.append(pretty_midi.Note(velocity=70, pitch=47, start=t + beat * 3.25, end=t + beat * 3.35))
            drums.notes.append(pretty_midi.Note(velocity=75, pitch=45, start=t + beat * 3.5, end=t + beat * 3.6))
            drums.notes.append(pretty_midi.Note(velocity=80, pitch=43, start=t + beat * 3.75, end=t + beat * 3.85))

        t += beat * 4

    midi.instruments.append(drums)
    midi.write(filename)
    print("🥁 Dynamic lofi drums saved:", filename)

create_lofi_drum_midi(bpm=80)