import matplotlib
import matplotlib.pylab as plt

import IPython.display as ipd

import sys
sys.path.append('waveglow/')
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser

import os

from pydub import AudioSegment

#command: python synthesize.py [tacotron2-model-path] [waveglow-model-path]

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050

if len(sys.argv) >= 2:
    checkpoint_path = sys.argv[1]
    print("using custom tacotron2 model: ", sys.argv[1])
else:
    try:
        f = open(os.path.join("outdir", "checkpoint_path.txt"), "r")
        checkpoint_path = f.read()
        print("using checkpoint tacotron2 model: ", checkpoint_path)
        f.close()
    except:
        print("error loading checkpoint model")
        exit()
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

if len(sys.argv) >= 3:
    waveglow_path = sys.argv[2]
    print("using custom waveglow model: ", sys.argv[2])
else:
    waveglow_path = 'models/waveglow_models/waveglow_256channels.pt'
    print("using pretrained waveglow model: ", 'models/waveglow_models/waveglow_256channels.pt')
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

evaluationsDirectory = "evaluations"
try:
    os.mkdir(evaluationsDirectory)
except OSError:
    print ("No creation of directory")
else:
    print ("Created the directory")

with open("evaluation_text.txt") as evaluation_file:
    lines = evaluation_file.read().splitlines()
    for line in lines:
        sequence = np.array(text_to_sequence(line, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()

        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        plot_data((mel_outputs.float().data.cpu().numpy()[0],
                mel_outputs_postnet.float().data.cpu().numpy()[0],
                alignments.float().data.cpu().numpy()[0].T))

        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

        line_filename = line.replace(".", "").replace("?", "~")

        audio_exp = ipd.Audio(audio.cpu().numpy(), rate=hparams.sampling_rate)
        audio_exp = AudioSegment(audio_exp.data, frame_rate=22050, sample_width=2, channels=1)
        audio_exp.export(evaluationsDirectory + "/" + line_filename + ".mp3", format="mp3", bitrate="64k")

        print("Unfiltered synthesis of " + line + " done!")

        audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

        audio_denoised_exp = ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
        audio_denoised_exp = AudioSegment(audio_denoised_exp.data, frame_rate=22050, sample_width=2, channels=1)
        audio_denoised_exp.export(evaluationsDirectory + "/" + line_filename + "-denoised.mp3", format="mp3", bitrate="64k")

        print("Denoised synthesis of " + line + " done!")

sys.exit()