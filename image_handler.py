import cv2
from tkinter import filedialog
from PIL import Image, ImageTk
import tkinter as tk

def load_image(canvas, edited_image_canvas):
    file_path = filedialog.askopenfilename()
    if file_path:
        img_cv = cv2.imread(file_path)
        display_image(img_cv, canvas, original=True)
        edited_image_canvas.delete("all")
        if edited_image_canvas is None:
            return img_cv
        return img_cv
    return None

def display_image(img, canvas, original=False):
    if img is None:
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    max_size = 500
    img_pil.thumbnail((max_size, max_size))
    img_tk = ImageTk.PhotoImage(img_pil)

    x_offset = (max_size - img_pil.width) // 2
    y_offset = (max_size - img_pil.height) // 2

    canvas.delete("all")
    canvas.image = img_tk
    canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
