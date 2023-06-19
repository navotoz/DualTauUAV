import argparse
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import tifffile as tifffile
from scipy.interpolate import interp1d
import tkinter as tk
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import numpy as np
from PIL import ImageTk
from PIL import Image
from tqdm import tqdm

from devices import Frame

HEIGHT_VIEWER = int(2 * 336)
WIDTH_VIEWER = int(2 * 256)


def save_figures(data: List[Frame]):
    def _plotter(lst, *, path, ylabel):
        plt.figure()
        plt.plot(list_time, lst)
        plt.grid()
        plt.xlabel('Time [Min]')
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    n_images = len(data)
    list_cnt_init = [p.cnt_init for p in data]
    list_cnt_final = [p.cnt_final for p in data]
    list_fpa = [p.fpa for p in data]
    list_housing = [p.housing for p in data]
    try:
        list_time = [p.position for p in data]
        list_time = [p.time if p is not None else None for p in list_time]
        n_timestamps = [p for p in range(n_images) if list_time[p] is not None]
        print(f'Missing {n_images - len(n_timestamps)} timestamps', flush=True)
        if len(n_timestamps) != n_images:
            f = interp1d(x=n_timestamps, y=list(filter(lambda x: x is not None, list_time)))
            list_time = f(np.arange(n_images))
        list_time = np.array(list_time)
        list_time = (list_time - list_time.min()) / 60
    except:
        list_time = np.arange(n_images)

    _plotter(list_cnt_init, path='cnt_init.png', ylabel='Counter Init')
    _plotter(list_cnt_final, path='cnt_final.png', ylabel='Counter Final')
    _plotter(list_fpa, path='fpa.png', ylabel='FPA [100C]')
    _plotter(list_housing, path='housing.png', ylabel='Housing [100C]')


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


def load(path) -> List[Image.Image]:
    try:
        return {k: v for k, v in np.load(str(path)).items()}
    except BadZipFile:
        raise RuntimeError(f"File {path} is corrupted.")
    except ValueError:
        return tifffile.imread(str(path))


def display(event):
    image = images[slider.get() - 1]
    size_root = (root.winfo_height(), root.winfo_width())
    size_canvas = (canvas.winfo_height(), canvas.winfo_width())
    if size_canvas != size_root:
        canvas.config(width=root.winfo_width(), height=root.winfo_height())
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


def save_key(event):
    images[slider.get()].save(path_to_files / f'{slider.get()}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_files', type=str, required=True)
    args = parser.parse_args()

    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', closer)
    root.title("Tau2 Images Viewer")
    root.option_add("*Font", "TkDefaultFont 14")
    root.geometry(f"{HEIGHT_VIEWER}x{WIDTH_VIEWER}")
    root.pack_propagate(0)

    path_to_files = Path(args.path_to_files)
    if path_to_files.is_file():
        data = load(path_to_files)
    elif not path_to_files.is_dir():
        raise NotADirectoryError(f'{path_to_files}')
    else:
        list_images = list(path_to_files.glob('*.npz'))
        if not list_images:
            list_images = list(path_to_files.glob('*.tif'))
        list_images.sort()
        if not list_images:
            raise FileNotFoundError(f"No files were found in {(path_to_files / 'images')}")
        # data = list(tqdm(map(load, list_images), total=len(list_images), desc='Load images'))
        with Pool(cpu_count()) as pool:
            data = list(tqdm(pool.imap(load, list_images), total=len(list_images), desc='Load images'))
        if not data:
            raise RuntimeError('Images were not loaded.')

        data_combined = {}
        cnt = 0
        for d in data:
            for k, v in d.items():
                if k == 'frames':
                    cnt += len(v)
                data_combined.setdefault(k, []).extend(v)
        data = data_combined
        assert len(data['frames']) == len(data['time']) == len(data['fpa'])

    # Get times
    try:
        times = np.stack(data['time'])
        indices = np.argsort(times)
    except KeyError:
        indices = np.arange(len(data['frames']))

    # Sort data
    data = {k: np.stack([v[i] for i in indices]) for k, v in data.items()}
    images = [normalize_image(p) for p in tqdm(data['frames'], desc='Normalize images')]

    # save_figures(data)
    slider = tk.Scale(root, from_=0, to=len(images), orient=tk.HORIZONTAL, resolution=1, command=display)
    slider.pack(side=tk.BOTTOM, fill=tk.BOTH)
    app = tk.Frame(root, bg='white')
    root.bind('<Left>', left_key)
    root.bind('<Right>', right_key)
    root.bind('<space>', save_key)
    app.pack()
    canvas = tk.Label(app)
    canvas.pack()
    root.mainloop()
