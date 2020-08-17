# Gesticulator demonstration

In this folder we provide two pretrained models (one with BERT and the other with FastText embedding) and the `demo.py` script.

Both audio and text input must be provided in order to generate new gestures.

For an example, run
```bash
python demo.py inputs/audio.wav inputs/text.json
```

Please note that the model was trained using time-annotated JSON text transcriptions from Google Speech-to-Text, but `demo.py` accepts plaintext transcriptions as well:

```bash
python demo.py inputs/audio.wav "This is a text transcription provided on the command line"
```
