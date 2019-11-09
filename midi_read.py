import mido
from music_synth import generate_mid
import os
import numpy as np


def get_data(author):
    songs = os.listdir('data/' + author)
    notes = []
    velocities = []
    times = []
    commands = []
    lengths = []

    for song in songs:
        mid = mido.MidiFile('data/' + author + '/' + song)
        length = 0
        for msg in mid:
            length += 1
            # print(msg)
            data = str(msg).split(" ")
            if data[0] == 'note_on' or data[0] == 'note_off':
                command = data[0]
                note = data[2][5:]
                velocity = data[3][9:]
                time = data[4][5:]

                notes.append(int(note))
                velocities.append(int(velocity))
                times.append(float(time))
                commands.append(command)
        lengths.append(length)

    assert len(notes) == len(velocities) == len(times)
    num_datos = len(notes)
    print("%i datos encontrados." % num_datos)

    # print("Generando midi...")
    # generate_mid(commands, notes, velocities, times)

    return commands, notes, velocities, times, num_datos

