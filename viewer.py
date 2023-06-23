import argparse
from zipfile import BadZipFile

import tifffile as tifffile
import tkinter as tk
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import ImageTk
from PIL import Image
from tqdm import tqdm

HEIGHT_VIEWER = int(2 * 336)
WIDTH_VIEWER = int(2 * 256)


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


def load_all_files_from_path(path: Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if path.is_file():
        data = load(path)
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

        data_combined = {}
        for d in data:
            for k, v in d.items():
                data_combined.setdefault(k, []).extend(v)
        indices = np.argsort(data_combined['time_ns'])
        for k, v in data_combined.items():
            data_combined[k] = np.stack(v)[indices]
        return data_combined


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
    images[slider.get()].save(path_____ / f'{slider.get()}.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', type=str, required=True)
    args = parser.parse_args()

    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', closer)
    root.title("Tau2 Images Viewer")
    root.option_add("*Font", "TkDefaultFont 14")
    root.geometry(f"{HEIGHT_VIEWER}x{WIDTH_VIEWER}")
    root.pack_propagate(0)

    data = load_all_files_from_path(args.files)

    # Normalize data
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
