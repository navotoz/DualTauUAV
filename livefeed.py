import os
if os.environ.get('DISPLAY','') == '':
    os.environ.__setitem__('DISPLAY', ':0.0')

import pyftdi.ftdi
from PIL import ImageTk

import tkinter as tk
from devices.Tau2Grabber import Tau2

from utils.misc import normalize_image

SCALE = 2.5
WIDTH_VIEWER = int(SCALE*336)
HEIGHT_VIEWER = int(SCALE*256)


def closer():
    try:
        camera_mono.__del__()
    except (RuntimeError, ValueError, NameError, KeyError, TypeError, AttributeError):
        pass
    try:
        camera_pan.__del__()
    except (RuntimeError, ValueError, NameError, KeyError, TypeError, AttributeError):
        pass
    try:
        root.destroy()
    except tk.TclError:
        pass


def th_viewer():
    try:
        frame_pan = camera_pan.grab()
    except (RuntimeError, ValueError, NameError, pyftdi.ftdi.FtdiError):
        pass
    try:
        frame_mono = camera_mono.grab()
    except (RuntimeError, ValueError, NameError, pyftdi.ftdi.FtdiError):
        pass
    if frame_mono is not None:
        image_tk = ImageTk.PhotoImage(normalize_image(frame_mono).resize((WIDTH_VIEWER, HEIGHT_VIEWER)))
        l_mono_label.image_tk = image_tk
        l_mono_label.configure(image=image_tk)
    if frame_pan is not None:
        image_tk = ImageTk.PhotoImage(normalize_image(frame_pan).resize((WIDTH_VIEWER, HEIGHT_VIEWER)))
        l_pan_label.image_tk = image_tk
        l_pan_label.configure(image=image_tk)
    root.after(ms=1000//30, func=th_viewer)


camera_pan = Tau2()
camera_pan.ffc_mode = 'auto'
camera_mono = Tau2()
camera_mono.ffc_mode = 'auto'

PAD_X = 10
PAD_Y = 10
root = tk.Tk()
root.protocol('WM_DELETE_WINDOW', closer)
root.title("Tau2 Livefeed")
root.geometry(f"{2 * WIDTH_VIEWER + 4 * PAD_X}x{HEIGHT_VIEWER + 2*PAD_Y}")

# Create the labels to display the frames
l_mono_label = tk.Label(root, width=WIDTH_VIEWER, height=HEIGHT_VIEWER)
l_mono_label.grid(row=0, column=0, padx=PAD_X, pady=PAD_Y)
l_pan_label = tk.Label(root, width=WIDTH_VIEWER, height=HEIGHT_VIEWER)
l_pan_label.grid(row=0, column=1, padx=PAD_X, pady=PAD_Y)

th_viewer()

root.mainloop()
