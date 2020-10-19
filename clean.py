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
import matplotlib
import timeit
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

sd_stream = None
file_path = None
device_in = 0
device_out = 1

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
        ratio_in = 16000/48000
        ratio_out = 48000/16000
        def callback(indata, outdata, frames, time, status):
            wav = indata[:,0]      
            wav = samplerate.resample(wav, ratio_in, 'sinc_best')
            # print(f'{len(wav)} -> {len(wav16)}')
            wav = normalize_wave_minmax(wav)        
            wav = pre_emphasize(wav, 0.95)        
            pwav = torch.FloatTensor(wav).view(1,1,-1)  
            g_wav, g_c = segan.generate(pwav)        
            g_wav = denormalize_wave_minmax(g_wav)  
            g_wav = samplerate.resample(g_wav, ratio_out, 'sinc_best')
            outdata[:,0] = g_wav

        try:
            file_train = 'train.opts' #os.path.join(os.path.dirname(sys.executable), 'train.opts')
            with open(file_train, 'r') as cfg_f:
                args = ArgParser(json.load(cfg_f))
            if hasattr(args, 'wsegan') and args.wsegan:
                segan = WSEGAN(args)     
            else:
                segan = SEGAN(args) 
                
            segan.G.load_pretrained(file_path, True)
            segan.G.eval()
            global sd_stream
            sd_stream = sd.Stream(device=(device_in, device_out),
                        samplerate=48000, blocksize=12000, #16000 - 1 sec
                        dtype='int16',
                        channels=1, callback=callback)
            sd_stream.start()
        except Exception as e:
            # parser.exit(type(e).__name__ + ': ' + str(e))
            button_start['state'] = tk.NORMAL
            button_stop['state']  = tk.DISABLED
            mb.showwarning("Exceptoin", str(e))


    def stop():    
        caption_status.set('Conversion stopped')
        button_start['state'] = tk.NORMAL
        button_stop['state']  = tk.DISABLED
        global sd_stream
        sd_stream.close()

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
        combobox_in_device.current(int(device_in))
        combobox_out_device.current(int(device_out))
        # combobox_in_device.set(devices[device_in])
        # combobox_out_device.set(devices[device_out])
    except Exception as e:
        device_in = 0
        device_out = 1
        combobox_in_device.current(int(device_in))
        combobox_out_device.current(int(device_out))

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





