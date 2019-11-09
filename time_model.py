import numpy as np
import random
from keras.layers import Dense, LSTM, Dropout, Activation, Bidirectional
from keras.models import Sequential
from keras.optimizers import RMSprop


def time_model(times, epochs):
    num_datos = len(times)

    maxlen = 30
    step = 2
    sequences = []
    next_notes = []
    for i in range(0, num_datos - maxlen, step):
        sequences.append(times[i: i + maxlen])
        next_notes.append(times[i + maxlen])
    x = np.array(sequences).astype('float32')
    x = x.reshape((len(sequences), 30, 1))
    y = np.array(next_notes).astype('float32')
    print(x.shape)
    print(y.shape)

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
        sentence = times[start_index: start_index + maxlen]
        generated += sentence
        for i in range(length):
            x_pred = np.array(sentence).astype('float32')
            # x_pred = np.expand_dims(x_pred, axis=0)
            x_pred = x_pred.reshape((1, 30, 1))

            preds = model.predict(x_pred, verbose=0)[0]
            next_note = sample(preds, diversity)

            generated += [next_note]
            sentence = sentence[1:] + [next_note]
        return generated

    notes_model = Sequential()
    notes_model.add(LSTM(256, input_shape=(maxlen, 1)))
    notes_model.add(Dropout(0.2))
    notes_model.add(Dense(1))
    notes_model.add(Activation('sigmoid'))
    optimizer = RMSprop(lr=0.01)
    notes_model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

    notes_model.fit(x, y, batch_size=64, epochs=epochs, verbose=0)

    music_gen = generate_music(3000, 0.3, notes_model)

    return music_gen
