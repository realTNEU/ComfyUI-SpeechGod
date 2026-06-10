Default voices for parameter-only character creation (no voice reference connected).

Both F5-TTS and Fish Speech are voice-CLONING models: they always speak in the
voice of a reference audio clip. When a character has no Voice Reference input,
Speech-God looks here for a seed voice, in this priority order:

  <gender>_<age>.wav     e.g.  female_child.wav, male_elderly.wav
  <gender>.wav           e.g.  male.wav
  neutral_adult.wav
  default.wav

Drop 5-15 second clean WAV clips (mono or stereo, any sample rate) of voices
you have rights to use. Good free sources: your own recordings, LibriVox
(public domain audiobooks), or CC0 voice datasets.

Age/gender/tone/energy parameters then reshape the seed voice via pitch,
tempo and dynamics so each character still sounds distinct.
