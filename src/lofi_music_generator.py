import glob
import os
import pickle
import numpy as np
import random
from music21 import converter, instrument, note, chord, stream, tempo

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def load_midi(path='/kaggle/input/dataset-aol-dl2/midi_songs/*.mid'):
    notes = []
    durations = []
    
    files = glob.glob(path)
    print(f"Found {len(files)} MIDI files")
    
    for file in files:
        try:
            midi = converter.parse(file)
            print(f"Parsing {file}")
            
            notes_to_parse = None
            
            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes
            
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.quarterLength)
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                    durations.append(element.quarterLength)
                    
        except Exception as e:
            print(f"Error parsing {file}: {e}")
            continue
    
    os.makedirs("data", exist_ok=True)
    with open('data/notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    with open('data/durations.pkl', 'wb') as filepath:
        pickle.dump(durations, filepath)
    
    print(f"\nTotal notes extracted: {len(notes)}")
    print(f"Unique notes: {len(set(notes))}")
    print(f"Sample notes: {notes[:20]}")
    
    return notes, durations



# DATA PREPARATION


def create_sequences(notes, sequence_length=64):
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    
    print(f"\nVocabulary size: {n_vocab}")
    print(f"Sequence length: {sequence_length}")
    
    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    
    network_input = []
    network_output = []
    
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        
        network_input.append([note_to_int[n] for n in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    print(f"Total patterns: {n_patterns}")
    
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    
    network_output = to_categorical(network_output, num_classes=n_vocab)
    
    return network_input, network_output, note_to_int, n_vocab



# MODEL ARCHITECTURE

def create_network(network_input, n_vocab):
    model = Sequential()


    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True,
        dropout=0.2
    ))

    model.add(GRU(
        512,
        dropout=0.2
    ))

    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(n_vocab, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    return model


# TRAINING

def train(model, network_input, network_output, epochs=100, batch_size=64):
    os.makedirs("weights", exist_ok=True)
    
    filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}.weights.h5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', 
        factor=0.5,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    callbacks_list = [checkpoint, reduce_lr]  
    
    history = model.fit(
        network_input,
        network_output,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    model.save("models/full_model.hdf5")

    return history


# GENEREATION

def top_k_sampling(preds, k=8, temperature=0.65):
    preds = np.asarray(preds).astype("float64")

    preds = preds ** (1.0 / temperature)

    top_indices = np.argsort(preds)[-k:]
    top_probs = preds[top_indices]

    top_probs = top_probs / np.sum(top_probs)

    return np.random.choice(top_indices, p=top_probs)


import random

def generate_notes(
    model,
    network_input,
    int_to_note,
    n_vocab,
    length=200,
    top_k=8,
    temperature=0.65
):


    start = np.random.randint(0, len(network_input) - 100)
    pattern = [int(p * n_vocab) for p in network_input[start]]

    prediction_output = []

    print(f"\n🎧 Generating {length} lo-fi notes...\n")

    for i in range(length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)[0]

        index = top_k_sampling(
            prediction,
            k=top_k,
            temperature=temperature
        )

        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]

        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{length}")

    print("\n Generation complete!")
    return prediction_output


# MIDI CREATION


def create_midi(prediction_output, filename="lofi_output.mid"):

    offset = 0
    output_notes = []

    STEP = 0.5          
    TAIL_SCALE = 1.6  

    for pattern in prediction_output:

        if '.' in pattern or pattern.isdigit():
            notes = []
            for n in pattern.split('.'):
                try:
                    new_note = note.Note(int(n))
                    new_note.quarterLength = STEP * TAIL_SCALE
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                except:
                    pass

            if notes:
                c = chord.Chord(notes)
                c.offset = offset
                output_notes.append(c)

        else:
            try:
                n = note.Note(pattern)
                n.offset = offset
                n.quarterLength = STEP * TAIL_SCALE
                n.storedInstrument = instrument.Piano()
                output_notes.append(n)
            except:
                pass

        
        offset += STEP

    midi_stream = stream.Stream(output_notes)

    midi_stream.insert(0, tempo.MetronomeMark(number=60))

    midi_stream.write("midi", fp=filename)
    print(f"Saved {filename}")




# MAIN EXECUTION

def main():
    notes, durations = load_midi()
    
    network_input, network_output, note_to_int, n_vocab = create_sequences(notes, sequence_length=64)
        
    int_to_note = {number: note for note, number in note_to_int.items()}
    with open('data/mappings.pkl', 'wb') as f:
            pickle.dump({'note_to_int': note_to_int, 'int_to_note': int_to_note, 'n_vocab': n_vocab}, f)
        
    model = create_network(network_input, n_vocab)
    model.summary()
        
    history = train(model, network_input, network_output, epochs=100, batch_size=64)
    
        # Step 5: Generate music
# print("\n[5/5] Generating music...")
    generated_notes = generate_notes(
        model,
        network_input,
        int_to_note,
        n_vocab,
        length=100,
        top_k=8,
        temperature=0.65
    )
        # Create MIDI files with different parameters
    create_midi(generated_notes, filename="lofi_output_balanced.mid")
    print("GENERATION COMPLETE!")
    
#     # Generate more variations

    

# UTILITY FUNCTIONS FOR LOADING PRETRAINED MODEL

def load_pretrained_and_generate(weights_file, length=200):
    """
    Load a pretrained model and generate music
    
    Args:
        weights_file: Path to saved weights
        length: Number of notes to generate
    """
    print("Loading saved data...")
    
    # Load notes and mappings
    with open('data/notes.pkl', 'rb') as f:
        notes = pickle.load(f)
    
    with open('data/mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        note_to_int = mappings['note_to_int']
        int_to_note = mappings['int_to_note']
        n_vocab = mappings['n_vocab']
    
    # Recreate sequences
    network_input, network_output, _, _ = create_sequences(notes, sequence_length=64)
    
    # Create and load model
    print("Loading model...")
    model = create_network(network_input, n_vocab)
    model.load_weights(weights_file)
    
    # Generate music
    print("Generating music...")
    generated_notes = generate_notes(
        model, network_input, int_to_note, n_vocab,
        length=length, top_k=6, temperature=0.85
    )
    
    create_midi(generated_notes, filename="lofi_from_pretrained.mid")
    print("Done!")


if __name__ == "__main__":
    main()