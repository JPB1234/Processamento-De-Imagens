import tkinter as tk
from tkinter import filedialog
from image_handler import load_image, display_image
from filters import low_pass, high_pass, low_pass_implemented, high_pass_implemented

img_cv = None

def set_img_cv(img):
    global img_cv
    img_cv = img

root = tk.Tk()
root.title("Image Processing App")

root.geometry("1085x550")
root.config(bg="#2e2e2e")

menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=lambda: set_img_cv(load_image(original_image_canvas)))
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)
filters_menu.add_command(label="Low Pass Filter", command=lambda: low_pass(img_cv, edited_image_canvas))
filters_menu.add_command(label="Low Pass Filter Implemented", command=lambda: low_pass_implemented(img_cv, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter", command=lambda: high_pass(img_cv, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter Implemented", command=lambda: high_pass_implemented(img_cv, edited_image_canvas))


original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
