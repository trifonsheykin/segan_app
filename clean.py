#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segan.models import *
from segan.datasets import *
import soundfile as sf
import sounddevice as sd
from scipy.io import wavfile
from torch.autograd import Variable
import numpy as np
import random
import librosa
import timeit
import json
import glob
import os
import sys
import time
import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog as fd 
from tkinter import messagebox as mb
import configparser
import multiprocessing
import samplerate
import queue
import threading

# exe file generation:
# pyinstaller -F --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" --onefile --noconsole clean.py
# app file generation:
# pyinstaller --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" --onefile --windowed --osx-bundle-identifier store.smartlocks.smartlock --add-binary='/System/Library/Frameworks/Tk.framework/Tk':'tk' --add-binary='/System/Library/Frameworks/Tcl.framework/Tcl':'tcl' clean.py
# codesign -s "Apple Development: Trifon Sheikin (VSF46QSYHU)" --deep clean.app


class ArgParser(object):
    def __init__(self, args):
        for k, v in args.items():
            setattr(self, k, v)

def denormalize_wave_minmax(x):
    return ((x - 1.) * (65636./2.)) + 32767.

input_stream = None
output_stream = None
file_path = None
audio_process = False
sound_thread = None
device_in = 0
device_out = 1
q_in = queue.Queue()
q_out = queue.Queue()
ratio_out = 48000/16000
data_in = np.zeros((8000,1), dtype=np.int16)
data_temp = np.zeros((8000,1), dtype=np.int16)
data_out = np.zeros((24000,1), dtype=np.float32)
i = np.iinfo(data_in.dtype)
abs_max = 2 ** (i.bits - 1)
offset = i.min + abs_max

def main():
    seed = 111
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #Create window
    window = tk.Tk() 
    window.title('Fluenta') 
    window.geometry('550x230') 

    #Read config file

    config = configparser.ConfigParser()
    file_config = 'config.conf' # os.path.join(os.path.dirname(sys.executable), 'config.conf')
    with open(file_config) as fp:
        config.readfp(fp) 
        global file_path  
        file_path = config['files']['weights']
        global device_in
        global device_out
        device_in = int(config['devices']['device_in'])
        device_out = int(config['devices']['device_out'])

    #Get available devices
    devices = []
    qd = sd.query_devices()
    for device in qd:
        devices.append(device['name'])

    caption_status = tk.StringVar()
    caption_status.set('Select weights file, specify input and output devices and press Start button')
    caption_path = tk.StringVar()
    caption_path.set(file_path)

    def start():
        caption_status.set('Conversion started, you may speak')
        button_start['state'] = tk.DISABLED
        button_stop['state']  = tk.NORMAL

        global device_in
        global device_out
        device_in = combobox_in_device.current()
        device_out = combobox_out_device.current()

        config['files']['weights'] = file_path
        config['devices']['device_in'] = str(device_in)
        config['devices']['device_out'] = str(device_out)
        with open(file_config, 'w') as configfile:
            config.write(configfile)

        segan = None


        def sound_processing():
            global audio_process
            audio_process = True
            while audio_process:
                item = q_in.get()
                print('PROCESSING:')
                wav = data_in[:,0] 
                wav = normalize_wave_minmax(wav) 
                wav = pre_emphasize(wav, 0.95)
                pwav = torch.FloatTensor(wav).view(1,1,-1) 
                g_wav, g_c = segan.generate(pwav)
                g_wav = denormalize_wave_minmax(g_wav)
                data_temp[:,0] = g_wav 
                audio = (data_temp.astype('float32') - offset) / abs_max
                audio = samplerate.resample(audio, ratio_out, 'sinc_best')
                if q_out.empty():
                    q_out.put(audio)
                    data_out[:] = audio

        def input_callback(indata, frames, time, status):
            print('INPUT:')
            data_in[:] = indata
            q_in.put(data_in)
        
        def output_callback(outdata, frames, time, status):
            print('OUTPUT:')
            if not q_out.empty():
                d = q_out.get()
            outdata[:] = data_out

        try:
            file_train = 'train.opts' #os.path.join(os.path.dirname(sys.executable), 'train.opts')
            with open(file_train, 'r') as cfg_f:
                args = ArgParser(json.load(cfg_f))
            segan = WSEGAN(args)     
            segan.G.load_pretrained(file_path, True)
            segan.G.eval()
            global input_stream
            global output_stream
            global sound_thread
            # sd_stream = sd.Stream(device=(device_in, device_out),
            #             samplerate=16000, blocksize=3200, #16000 - 1 sec
            #             dtype='int16',
            #             channels=1, callback=callback)
            input_stream = sd.InputStream(device=device_in, channels=1, dtype='int16',
                            blocksize=8000, #16000 - 1 sec
                            samplerate=16000, 
                            callback=input_callback)
            
            output_stream = sd.OutputStream(device=device_out, channels=1, dtype='float32', 
                            blocksize=24000,
                            samplerate=48000, 
                            callback=output_callback)
            
            sound_thread = threading.Thread(target=sound_processing)

            sound_thread.start()
            input_stream.start()
            output_stream.start()

        except Exception as e:
            # parser.exit(type(e).__name__ + ': ' + str(e))
            button_start['state'] = tk.NORMAL
            button_stop['state']  = tk.DISABLED
            mb.showwarning("Exceptoin", str(e))


    def stop():    
        caption_status.set('Conversion stopped')
        button_start['state'] = tk.NORMAL
        button_stop['state']  = tk.DISABLED
        global input_stream
        global output_stream
        global audio_process
        global sound_thread
        audio_process = False
        q_in.put(data_in)
        sound_thread.join()
        input_stream.close()
        output_stream.close()
        print('CONVERSION STOPED')

    def open_file(): 
        global file_path
        file_path = fd.askopenfilename(filetypes =[("Weights file", "*.ckpt")]) 
        caption_path.set(file_path)

    tk.Label(window, text='Generator weights file:').place(x=7, y=3)
    edit_path = tk.Entry(window, textvariable=caption_path, width=51)
    edit_path.place(x=5, y=26)
    button_open  = tk.Button(text='Open',command=open_file)
    button_open.place(x=475, y=24)

    tk.Label(window, text='Input device:').place(x=7, y=53)
    combobox_in_device = ttk.Combobox(window, values=devices, width=57)
    combobox_in_device.place(x=7, y=75)

    tk.Label(window, text='Output device:').place(x=7, y=100)
    combobox_out_device = ttk.Combobox(window, values=devices, width=57)
    combobox_out_device.place(x=7, y=123)

    try:
        # check we have input/output channels, revert to defaults if not
        if qd[device_in]['max_input_channels']==0 or \
           qd[device_out]['max_output_channels']==0 or max(device_in, device_out) >= len(qd):
            device_in, device_out = sd.default.device
        combobox_in_device.current(device_in)
        combobox_out_device.current(device_out)
    except Exception as e:
        device_in, device_out = 0, 1
        combobox_in_device.current(device_in)
        combobox_out_device.current(device_out)

    tk.Label(window, text='Whisper conversion:').place(x=7, y=150)
    button_start = tk.Button(text='Start',command=start, width=23, state=tk.NORMAL)
    button_start.place(x=20, y=173)
    button_stop  = tk.Button(text='Stop',command=stop, width=23, state=tk.DISABLED)
    button_stop.place(x=285, y=173)

    label_status = tk.Label(window, textvariable=caption_status).place(x=7, y=205)

    window.mainloop() 

if __name__ == '__main__':

# Pyinstaller fix
    multiprocessing.freeze_support()

    main()

# --------------------------------------------------------------------------------------------------------------------------





