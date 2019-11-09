from mido import Message, MidiFile, MidiTrack, second2tick, MetaMessage
import os


def generate_mid(name, commands, notes, velocities, times):
    dir_mid = os.getcwd() + '\\generated\\' + name + '.mid'

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(Message('program_change', program=0, time=0))
    for comm, note, vel in zip(commands, notes, velocities):
        t = int(second2tick(second=0.1, tempo=500000, ticks_per_beat=400))
        track.append(Message(comm, note=note, velocity=vel, time=t))

    mid.save(dir_mid)

    print("Midi saved in ", dir_mid)
