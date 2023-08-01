import argparse
from zipfile import BadZipFile
import re

import tifffile as tifffile
import tkinter as tk
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import ImageTk
from PIL import Image
from tqdm import tqdm

from notebooks.optim_correspondance import ensure_counts_on_both_files

SCALE = 3
WIDTH_VIEWER = int(SCALE * 336)
HEIGHT_VIEWER = int(SCALE * 256)
relations_height_to_width = HEIGHT_VIEWER / WIDTH_VIEWER
low_frame = -1
high_frame = -1
data_to_save_mono = {}
data_to_save_pan = {}


def normalize_image(image: np.ndarray) -> Image.Image:
    image = image.astype('float32')
    mask = image > 0
    image[~mask] = float('inf')
    image -= image.min(-1, keepdims=True).min(-2, keepdims=True)
    image[~mask] = -float('inf')
    image = image / image.max(-1, keepdims=True).max(-2, keepdims=True)
    image[~mask] = 0
    image *= 255
    if len(image.shape) > 2:
        return [Image.fromarray(p.astype('uint8')) for p in image]
    return Image.fromarray(image.astype('uint8'))


def validate_input(input):
    # Check if the input matches a regular expression for numbers
    if re.match("^[0-9]+$", input):
        return True
    else:
        return False


def load(path) -> List[Image.Image]:
    try:
        return {k: v for k, v in np.load(str(path)).items()}
    except (BadZipFile, EOFError):
        return None
    except ValueError:
        return tifffile.imread(str(path))


def load_all_files_from_path(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if path.is_file():
        return load(path)
    elif not path.is_dir():
        raise NotADirectoryError(f'{path}')
    else:
        list_images = list(path.glob('*.npz'))
        if not list_images:
            list_images = list(path.glob('*.tif'))
        list_images.sort()
        if not list_images:
            raise FileNotFoundError(f"No files were found in {(path / 'images')}")
        # data = list(tqdm(map(load, list_images), total=len(list_images), desc='Load images'))
        with Pool(cpu_count()) as pool:
            data = list(tqdm(pool.imap(load, list_images), total=len(list_images), desc='Load images'))
        if not data:
            raise RuntimeError('Images were not loaded.')
        data = list(filter(lambda x: x is not None, data))

        data_combined = {}
        for d in data:
            for k, v in d.items():
                data_combined.setdefault(k, []).extend(v)
        indices = np.argsort(data_combined['counter'])
        for k, v in data_combined.items():
            data_combined[k] = np.stack(v)[indices]
        return data_combined


def display(event):
    image = images[slider.get() - 1]
    height_label = input_entry.winfo_height()
    height_entry = input_entry.winfo_height()
    size_root = (root.winfo_height(), root.winfo_width())
    size_canvas = (canvas.winfo_height(), canvas.winfo_width())
    if size_canvas != size_root:
        canvas_height = root.winfo_height() - height_label - height_entry
        canvas_width = int(canvas_height / relations_height_to_width)
        canvas.config(width=canvas_width, height=canvas_height)
    image_tk = ImageTk.PhotoImage(image.resize(reversed(size_canvas)))
    canvas.image_tk = image_tk
    canvas.configure(image=image_tk)


def closer():
    try:
        root.destroy()
    except tk.TclError:
        pass


def left_key(event):
    slider.set(max(slider.get() - 1, 0))


def right_key(event):
    slider.set(min(slider.get() + 1, len(images)))


def mark_low_frame(event):
    global low_frame
    low_frame = slider.get()
    print(f'Lowest frame number: {low_frame}')


def mark_high_frame(event):
    global high_frame
    high_frame = slider.get()
    print(f'Highest frame number: {high_frame}')


def append_to_data(event):
    global low_frame, high_frame, data_to_save_pan, data_to_save_mono
    if low_frame == -1:
        low_frame = 0
    if high_frame == -1:
        high_frame = len(images)
    height_of_frames = int(input_entry.get())
    pan = {k: v[low_frame:high_frame] for k, v in data_pan.items()}
    pan['height'] = np.array([height_of_frames] * len(pan['frames']), dtype='uint16')
    for k, v in pan.items():
        data_to_save_pan.setdefault(k, []).append(v)
    mono = {k: v[low_frame:high_frame] for k, v in data_mono.items()}
    mono['height'] = np.array([height_of_frames] * len(mono['frames']), dtype='uint16')
    for k, v in mono.items():
        data_to_save_mono.setdefault(k, []).append(v)
    print(f'Appended {low_frame}:{high_frame} with {height_of_frames} to data.')


def save_single_file(event):
    global low_frame, high_frame
    if low_frame == -1:
        low_frame = 0
    if high_frame == -1:
        high_frame = len(images)
    for k, v in data_to_save_pan.items():
        data_to_save_pan[k] = np.concatenate(v)
    for k, v in data_to_save_mono.items():
        data_to_save_mono[k] = np.concatenate(v)
    np.savez(Path(args.files) / 'mono', **data_to_save_mono)
    np.savez(Path(args.files) / 'pan', **data_to_save_pan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, required=True)
    args = parser.parse_args()

    # Load both the pan and mono images
    path_to_all_files = Path(args.files)
    if path_to_all_files.is_file():
        data_pan = load(path_to_all_files)
    else:
        path_pan = path_to_all_files / 'pan'
        if not path_pan.exists():
            raise FileNotFoundError(f'{path_pan} does not exist.')
        data_pan = load_all_files_from_path(path_pan)
        path_mono = path_to_all_files / 'mono'
        if not path_mono.exists():
            raise FileNotFoundError(f'{path_mono} does not exist.')
        data_mono = load_all_files_from_path(path_mono)

        # Compare counter values
        data_mono, data_pan = ensure_counts_on_both_files(src=data_mono, dest=data_pan)

    # Normalize pan data for display
    images = [normalize_image(p) for p in tqdm(data_pan['frames'], desc='Normalize images')]

    # Construct GUI
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', closer)
    root.title("Tau2 Images Viewer")
    root.option_add("*Font", "TkDefaultFont 14")
    root.geometry(f"{WIDTH_VIEWER}x{HEIGHT_VIEWER}")
    root.pack_propagate(0)

    # Add user input
    _entry_label = tk.Label(root)
    _entry_label.pack(side=tk.TOP, fill=tk.X)
    input_label = tk.Label(_entry_label, text="Enter height of frames:")
    input_label.pack(side=tk.LEFT, padx=10, pady=10)
    validate_command = root.register(validate_input)
    input_entry = tk.Entry(_entry_label, validate='key', validatecommand=(validate_command, "%P"))
    input_entry.pack(side=tk.RIGHT, padx=10, pady=10)

    # Save data to file
    slider = tk.Scale(root, from_=0, to=len(images), orient=tk.HORIZONTAL, resolution=1, command=display)
    slider.pack(side=tk.BOTTOM, fill=tk.BOTH)
    app = tk.Frame(root, bg='white')
    root.bind('<Left>', left_key)
    root.bind('<Right>', right_key)

    # Pressing the 'a' key marks the lowest frame number
    # Pressing the 'd' key marks the highest frame number
    root.bind('<a>', mark_low_frame)
    root.bind('<d>', mark_high_frame)
    root.bind('<s>', save_single_file)
    root.bind('<w>', append_to_data)

    app.pack()
    canvas = tk.Label(app)
    canvas.pack()

    print('\nInstructions:')
    print('Press the left and right arrow keys to navigate through the images.')
    print('Press the \'a\' key to mark the lowest frame number.')
    print('Press the \'d\' key to mark the highest frame number.')
    print('Input the height of the frames in the entry box.')
    print('Press the \'w\' key to append the marked frames to the data.')
    print('Press the \'s\' key to save the data to a file.')

    root.mainloop()
