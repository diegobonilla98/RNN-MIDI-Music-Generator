from midi_read import get_data
from notes_model import notes_model
from command_model import command_model
from time_model import time_model
from velocity_model import velocity_model
from music_synth import generate_mid
import logging
logging.getLogger('tensorflow').disabled = True


commands, notes, velocities, times, _ = get_data('beethoven')
epochs = 5

# time_model = time_model(times, epochs)
print("Generando notas...")
notes_model = notes_model(notes, epochs)
print("Recogiendo propiedades de nota...")
velocity_model = velocity_model(velocities, epochs)
print("Aprendiendo a tocar el piano...")
command_model = command_model(commands, epochs)

print("Cancion hecha!")
generate_mid(name='artificial_beethoven', commands=command_model, notes=notes_model,
             velocities=velocity_model, times=1)

