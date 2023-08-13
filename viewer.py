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
data_to_save = {}


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
    if re.match("^[0-9]+$", input) or input == '':
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
        if 'counter' in data_combined:
            indices = np.argsort(data_combined['counter'])
        elif 'time_ns_start' in data_combined:
            indices = np.argsort(data_combined['time_ns_start'])
        elif 'time_ns' in data_combined:
            indices = np.argsort(data_combined['time_ns'])
        for k, v in data_combined.items():
            data_combined[k] = np.stack(v)[indices]
        return data_combined


def display(event):
    image = images[slider.get() - 1]
    height_label = label_input_height.winfo_height()
    height_entry = input_height.winfo_height()
    path_label = label_input_path.winfo_height()
    path_entry = input_path.winfo_height()
    size_root = (root.winfo_height(), root.winfo_width())
    size_canvas = (canvas.winfo_height(), canvas.winfo_width())
    if size_canvas != size_root:
        canvas_height = root.winfo_height() - height_label - height_entry - path_label - path_entry
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
    global low_frame, high_frame, data_to_save
    if low_frame == -1:
        low_frame = 0
    if high_frame == -1:
        high_frame = len(images)
    height_of_frames = int(input_height.get())
    pan = {k: v[low_frame:high_frame] for k, v in data_pan.items()}
    pan['height'] = np.array([height_of_frames] * len(pan['frames']), dtype='uint16')
    for k, v in pan.items():
        data_to_save.setdefault(k, []).append(v)
    print(f'Appended {low_frame}:{high_frame} with {height_of_frames} to data.')


def save_single_file(event):
    global low_frame, high_frame, path_to_all_files
    if low_frame == -1:
        low_frame = 0
    if high_frame == -1:
        high_frame = len(images)
    for k, v in data_to_save.items():
        data_to_save[k] = np.concatenate(v)
    if path_to_all_files.is_file():
        path_to_all_files = path_to_all_files.parent
    path_to_folder = path_to_all_files / 'results'
    path_to_folder.mkdir(exist_ok=True, parents=True)
    path_to_save = (path_to_folder / input_path.get()).with_suffix('.npz')
    idx = 1
    while path_to_save.exists():
        path_to_save = (path_to_folder / f'{input_path.get()}_{idx}').with_suffix('.npz')
        idx += 1
    np.savez(path_to_save, **data_to_save)


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
        if path_pan.with_suffix('.npz').exists():
            path_pan = path_pan.with_suffix('.npz')
        if not path_pan.exists():
            raise FileNotFoundError(f'{path_pan} does not exist.')
        data_pan = load_all_files_from_path(path_pan)

    # Normalize pan data for display
    images = [normalize_image(p) for p in tqdm(data_pan['frames'], desc='Normalize images')]

    # Construct GUI
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', closer)
    root.title("Tau2 Images Viewer")
    root.option_add("*Font", "TkDefaultFont 14")
    root.geometry(f"{WIDTH_VIEWER}x{HEIGHT_VIEWER}")
    root.pack_propagate(0)

    # Add user input - path to save data
    _label_path = tk.Label(root)
    _label_path.pack(side=tk.TOP, fill=tk.X)
    label_input_path = tk.Label(_label_path, text="Enter name of file to save data:")
    label_input_path.pack(side=tk.LEFT, padx=10, pady=10)
    input_path = tk.Entry(_label_path)
    input_path.pack(side=tk.RIGHT, padx=10, pady=10)

    # Add user input - height
    _label_height = tk.Label(root)
    _label_height.pack(side=tk.TOP, fill=tk.X)
    label_input_height = tk.Label(_label_height, text="Enter height of frames:")
    label_input_height.pack(side=tk.LEFT, padx=10, pady=10)
    validate_command = root.register(validate_input)
    input_height = tk.Entry(_label_height, validate='key', validatecommand=(validate_command, "%P"))
    input_height.pack(side=tk.RIGHT, padx=10, pady=10)

    # Save data to file
    slider = tk.Scale(root, from_=0, to=len(images), orient=tk.HORIZONTAL, resolution=1, command=display)
    slider.pack(side=tk.BOTTOM, fill=tk.BOTH)
    app = tk.Frame(root, bg='white')
    root.bind('<Left>', left_key)
    root.bind('<Right>', right_key)

    # Pressing the 'a' key marks the lowest frame number
    # Pressing the 'd' key marks the highest frame number
    root.bind('<Control-a>', mark_low_frame)
    root.bind('<Control-d>', mark_high_frame)
    root.bind('<Control-s>', save_single_file)
    root.bind('<Control-w>', append_to_data)

    app.pack()
    canvas = tk.Label(app)
    canvas.pack()

    print('\nInstructions:')
    print('Press the left and right arrow keys to navigate through the images.')
    print('Press the Ctrl + a key to mark the lowest frame number.')
    print('Press the Ctrl + d key to mark the highest frame number.')
    print('Input the height of the frames in the entry box.')
    print('Press the Ctrl + w key to append the marked frames to the data.')
    print('Input the name of the file to save in the entry box.')
    print('Press the Ctrl + s key to save the data to a file.')

    root.mainloop()
