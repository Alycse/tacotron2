import sys

import os
import shutil
import soundfile as sf
import ffmpy
from pydub import AudioSegment

import librosa
import wavio

codename = sys.argv[1]
codename_capped = codename.upper()

if len(sys.argv) < 2:
    print("Error: No name")
    exit()

generatedDirectoryPath = codename
generatedWavsDirectoryPath = codename + "/wavs"
try:
    os.mkdir(generatedDirectoryPath)
    os.mkdir(generatedWavsDirectoryPath)
except OSError:
    print ("No creation of directory")
else:
    print ("Created the directory")

metadata = ""
metadatatrain = ""
metadatatest = ""
metadataval = ""
metadatawaveglowtrain = ""
metadatawaveglowtest = ""

count = 0
seconds = 0
rawAudioDirectoryPath = os.path.dirname(os.path.realpath(__file__))
for entry in os.listdir(rawAudioDirectoryPath):
    if os.path.isfile(os.path.join(rawAudioDirectoryPath, entry)):
        if "wav" in entry or "WAV" in entry:

            f = sf.SoundFile(entry)
            print('seconds = {}'.format(len(f) / f.samplerate))
            seconds += len(f) / f.samplerate

            newFilename = codename_capped "-" + str(count);

            newListFilename = codename + "/wavs/" + codename_capped + "-" + str(count) + ".wav";
            newWaveListFilename = codename + "/wavs/" + codename_capped + "-" + str(count) + ".wav";

            y, s = librosa.load(entry, sr=22050)
            wavio.write(generatedWavsDirectoryPath + "/" + newFilename + ".wav", y, 22050, sampwidth=2)

            entry = entry.replace("’", "'")
            entry = entry.replace("“", "\"")

            count += 1
            
            if("~" in entry):
                metadata += newFilename + "|" + entry.replace(".wav", "").replace("~", "?") + "|" + entry.replace(".wav", "").replace("~", "?") + "\n"
                if count <= 5:
                    metadatatest += newListFilename + "|" + entry.replace(".wav", "").replace("~", "?") + "\n"
                    metadatawaveglowtest += newWaveListFilename + "\n"
                elif count <= 10:
                    metadataval += newListFilename + "|" + entry.replace(".wav", "").replace("~", "?") + "\n"
                    metadatawaveglowtest += newWaveListFilename + "\n"
                else:
                    metadatatrain += newListFilename + "|" + entry.replace(".wav", "").replace("~", "?") + "\n"
                    metadatawaveglowtrain += newWaveListFilename + "\n"
            else:
                metadata += newFilename + "|" + entry.replace(".wav", ".") + "|" + entry.replace(".wav", ".") + "\n"
                if count <= 5:
                    metadatatest += newListFilename + "|" + entry.replace(".wav", ".") + "\n"
                    metadatawaveglowtest += newWaveListFilename + "\n"
                elif count <= 10:
                    metadataval += newListFilename + "|" + entry.replace(".wav", ".") + "\n"
                    metadatawaveglowtest += newWaveListFilename + "\n"
                else:
                    metadatatrain += newListFilename + "|" + entry.replace(".wav", ".") + "\n"
                    metadatawaveglowtrain += newWaveListFilename + "\n"

print("total seconds: " + str(seconds) + ", total minutes: " + str(seconds/60))

file = open(generatedDirectoryPath + "/metadata.csv", "w")
file.write(metadata)
file.close()

filetrain = open(generatedDirectoryPath + "/" + codename + "_audio_text_train_filelist.txt", "w")
filetrain.write(metadatatrain)
filetrain.close()

filetest = open(generatedDirectoryPath + "/" + codename + "_audio_text_test_filelist.txt", "w")
filetest.write(metadatatest)
filetest.close()

fileval = open(generatedDirectoryPath + "/" + codename + "_audio_text_val_filelist.txt", "w")
fileval.write(metadataval)
fileval.close()

filewaveglowtrain = open(generatedDirectoryPath + "/train_files.txt", "w")
filewaveglowtrain.write(metadatawaveglowtrain)
filewaveglowtrain.close()

filewaveglowtest = open(generatedDirectoryPath + "/test_files.txt", "w")
filewaveglowtest.write(metadatawaveglowtest)
filewaveglowtest.close()