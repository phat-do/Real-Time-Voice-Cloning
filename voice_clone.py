# adapted for convenience on Google Colab

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
from IPython.display import Audio

def voice_clone(embed,
                synthesizer_path="saved_models/default/synthesizer.pt",
                vocoder_path="saved_models/default/vocoder.pt",
                cpu=False,
		no_sound=False):

    ## Load the models one by one.
    ensure_default_models(Path("saved_models"))
    synthesizer = Synthesizer(Path(synthesizer_path))
    vocoder.load_model(Path(vocoder_path))

    num_generated = 0
    while True:
        try:
            text = input("Write a sentence (+-20 words) to be synthesized:\n")
            texts = [text]
            embeds = [embed]
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            generated_wav = vocoder.infer_waveform(spec)
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
            generated_wav = encoder.preprocess_wav(generated_wav)

            # Play the audio (non-blocking)
            if not no_sound:
                try: display(Audio(generated_wav, rate=synthesizer.sample_rate))
                except:
                    print("\nCaught exception: %s" % repr(e))
                    print("Continuing without audio playback. Suppress this message with the \"--no_sound\" flag.\n")

            # Save it on the disk
            filename = "demo_output_%02d.wav" % num_generated
            # print(generated_wav.dtype)
            sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
