from midi_read import get_data
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop
import logging

logging.getLogger('tensorflow').disabled = True


def notes_model(notes, epochs):
    num_datos = len(notes)
    note_register = sorted(list(set(notes)))
    # print(note_register)
    print("%i notas en total." % len(note_register))

    note_indices = dict((c, i) for i, c in enumerate(note_register))
    indices_note = dict((i, c) for i, c in enumerate(note_register))

    maxlen = 20
    step = 1
    sequences = []
    next_notes = []
    for i in range(0, num_datos - maxlen, step):
        sequences.append(notes[i: i + maxlen])
        next_notes.append(notes[i + maxlen])

    x = np.zeros((len(sequences), maxlen, len(note_register)), dtype=np.bool)
    y = np.zeros((len(sequences), len(note_register)), dtype=np.bool)

    for i, sequence in enumerate(sequences):
        for t, note in enumerate(sequence):
            x[i, t, note_indices[note]] = 1
        y[i, note_indices[next_notes[i]]] = 1

    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_music(length, diversity, model):
        start_index = random.randint(0, num_datos - maxlen - 1)
        generated = []
        sentence = notes[start_index: start_index + maxlen]
        generated += sentence
        for i in range(length):
            x_pred = np.zeros((1, maxlen, len(note_register)), dtype=np.bool)
            for t, note in enumerate(sentence):
                x_pred[0, t, note_indices[note]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_note = indices_note[next_index]

            generated += [next_note]
            sentence = sentence[1:] + [next_note]
        return generated

    notes_model = Sequential()
    notes_model.add(LSTM(128, input_shape=(maxlen, len(note_register))))
    notes_model.add(Dropout(0.2))
    notes_model.add(Dense(len(note_register)))
    notes_model.add(Activation('softmax'))
    optimizer = RMSprop(lr=0.01)
    notes_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    notes_model.fit(x, y, batch_size=64, epochs=epochs, verbose=0)

    music_gen = generate_music(3000, 0.5, notes_model)

    return music_gen
