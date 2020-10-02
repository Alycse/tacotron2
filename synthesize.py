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

from pydub import AudioSegment

#command: python synthesize.py [tacotron2-model-path] [waveglow-model-path]

if len(sys.argv) < 3:
    print("Argument list invalid")
    exit()

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom', 
                       interpolation='none')

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = sys.argv[1]
model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

if len(sys.argv) < 2:
    waveglow_path = 'models/waveglow_models/waveglow_256channels.pt'
else:
    waveglow_path = sys.argv[2]
waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)

evaluation_file = open("evaluation_text.txt", "r")
text = evaluation_file.read()
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(
    torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
plot_data((mel_outputs.float().data.cpu().numpy()[0],
           mel_outputs_postnet.float().data.cpu().numpy()[0],
           alignments.float().data.cpu().numpy()[0].T))

with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

audio_exp = ipd.Audio(audio.cpu().numpy(), rate=hparams.sampling_rate)
audio_exp = AudioSegment(audio_exp.data, frame_rate=22050, sample_width=2, channels=1)
audio_exp.export("evaluation.mp3", format="mp3", bitrate="64k")

print("Unfiltered synthesis done!")

audio_denoised = denoiser(audio, strength=0.01)[:, 0]
ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)

audio_denoised_exp = ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
audio_denoised_exp = AudioSegment(audio_denoised_exp.data, frame_rate=22050, sample_width=2, channels=1)
audio_denoised_exp.export("evaluation-denoised.mp3", format="mp3", bitrate="64k")

print("Denoised synthesis done!")

sys.exit()