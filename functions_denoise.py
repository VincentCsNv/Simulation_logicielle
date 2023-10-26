import numpy as np
import soundfile as sf
import IPython 
import random as rd
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf


##############################################    
# I. Gathered DATA:
##############################################

def get_files_from_directories(path_data_folder = "./data"):
#Not perfect because selecting only 1 folder.
    list_audio_files = [path_data_folder]
    while os.path.isdir(list_audio_files[0]):
        origin_dir = list_audio_files[0]
        list_files_dir = os.listdir(origin_dir)
        list_audio_files = [origin_dir + "/" + list_files_dir[i] for i in range(len(list_files_dir))]
    print(f"We found {len(list_audio_files)} files")
    return list_audio_files

##############################################    
#II. Creating a noisy_signal:
##############################################

def random_noise_start(noise):
# UN RANDOM DANS CETTE FONCTION
    noise_resize = np.resize(noise, 2 * len(noise))
    random_start = rd.randint(0,len(noise))
    new_noise = noise_resize[random_start : random_start + len(noise)]
    return new_noise

def create_superpose_noise( noise , random_nb_superposition = True, nb_superposition = None):
# UN RANDOM DANS CETTE FONCTION
    if random_nb_superposition:
        nb_superposition = rd.randint(1,4)
    print(f"We used a superposition of {nb_superposition} noise(s)")

    superpose_noise = random_noise_start(noise)
    for i in range(nb_superposition-1):
        superpose_noise += random_noise_start(noise)
    superpose_noise = (1/nb_superposition)*superpose_noise
    return superpose_noise

def compute_power(signal):
    signal_stft = librosa.stft(signal)
    pw_signal = np.sum(np.abs(signal_stft**2))
    return pw_signal

def compute_rsb(signal, noise):
    pw_signal = compute_power(signal)
    pw_noise = compute_power(noise)
    rsb = pw_signal/pw_noise
    return rsb

def create_noisy_signal(clean_audio,noise, RSB = 0.7, random_nb_superposition = True, nb_superposition = None):
    superpose_noise = create_superpose_noise(noise,random_nb_superposition = random_nb_superposition, nb_superposition = nb_superposition)
    RSB_default = compute_rsb(clean_audio,superpose_noise)
    alpha = np.sqrt(RSB_default/RSB)
    noisy_signal = clean_audio + alpha*np.resize(superpose_noise, len(clean_audio))
    print(f"We have a rsb of {compute_rsb(clean_audio,alpha*superpose_noise)} for a noise's attenuation equals to alpha = {alpha}" )
    return noisy_signal, superpose_noise

##############################################    
#III. Preprocess the signals:
##############################################
def compute_abs_squared_stft(signal):
    signal_stft = librosa.stft(signal)
    signal_stft_abs = np.abs(signal_stft)**2
    return signal_stft_abs

def display_spectogramm(signal):
    #signal_stft_abs = compute_abs_squared_stft(signal)
    signal_stft_abs = signal
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(signal_stft_abs,ref=np.max),y_axis='log', x_axis='time', ax=ax)
    ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")

def get_spectro_signal(signal):
    signal_abs = compute_abs_squared_stft(signal)
    img = librosa.amplitude_to_db(signal_abs,ref=np.max)
    #img = librosa.display.specshow(librosa.amplitude_to_db(signal_abs,ref=np.max),y_axis='log', x_axis='time', ax=ax)
    return img

def get_best_mask(signal,noise):
    signal_stft_abs = compute_abs_squared_stft(signal)
    noise_stft_abs = compute_abs_squared_stft(noise)
    mask = np.zeros(np.shape(signal_stft_abs))
    for i in range (np.shape(signal_stft_abs)[0]):
        for j in range(np.shape(signal_stft_abs)[1]):
            if (signal_stft_abs[i,j] > noise_stft_abs[i,j]):
                mask[i,j] = 1
    return mask

def display_spectogramm_comparison(signal,noisy_signal,noise):
# DIDNT WORKED 
#problem au niveau de la definition de ax. (à la base fig, ax = plt.subplots(), mais là en voulant utiliser plt.subplot je suis bloqué.)
    signal_stft_abs_squared = compute_abs_squared_stft(signal)
    noisy_signal_stft_abs_squared = compute_abs_squared_stft(noisy_signal)
    noise_stft_abs_squared = compute_abs_squared_stft(noise)
    mask = get_best_mask(signal,noise)
    best_denoise_signal = signal_stft_abs_squared * mask
    fig , ax = plt.subplots(1,3,1)
    #fig ,ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(signal_stft_abs_squared,ref=np.max),y_axis='log', x_axis='time', ax=ax)
    #ax.set_title('Power spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
