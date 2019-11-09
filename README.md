# Music-Generator
A better version of my old Music Generator.

Now using MIDI so various notes can be played at one for different time.
The ANN still using RNN (LSMT) to predict new data given the last x values. I used 3 separate networks to learn the notes, speed and when the note stopped playing.
The input were Beethoven songs in .mid format I found online. The output was inpressive (see post on my Linkedln: https://www.linkedin.com/in/diego-bonilla-salvador/)

For future versions of this project I would like to use Disentangled GANs for fine tunning and more advanced architecture.


