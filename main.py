import tkinter as tk
from image_handler import load_image, display_image
from filters2 import *

img_cv = None
processed_img_cv = None  

def set_img_cv(img):
    global img_cv, processed_img_cv
    img_cv = img
    processed_img_cv = img.copy()  
    display_image(img_cv, original_image_canvas, original=True)

def apply_filter(filter_function, canvas, *args, **kwargs):
    global processed_img_cv
    if processed_img_cv is None:
        return
    processed_img_cv = filter_function(processed_img_cv, *args, **kwargs)  
    display_image(processed_img_cv, canvas, original=False)  

def high_pass_slider(root, canvas):
    def on_slider_change(value):
        global processed_img_cv
        if processed_img_cv is None:
            return
        kernel_value = int(value)
        processed_img_cv = high_pass_laplacian(processed_img_cv, kernel_value)  # Atualiza a imagem processada
        display_image(processed_img_cv, canvas, original=False)

    slider = tk.Scale(root, from_=1, to=15, orient=tk.HORIZONTAL, label="Kernel", command=on_slider_change)
    slider.grid(row=0, column=0, padx=50, pady=50)
    slider.set(3)

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
filters_menu.add_command(label="Low Pass Filter", command=lambda: apply_filter(low_pass, edited_image_canvas))
filters_menu.add_command(label="Low Pass Filter Gaussian", command=lambda: apply_filter(low_pass_gaussian, edited_image_canvas))
filters_menu.add_command(label="Low Pass Filter Media", command=lambda: apply_filter(low_pass_media, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter", command=lambda: apply_filter(high_pass, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter Laplacian", command=lambda: high_pass_slider(root, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter Sobel", command=lambda: apply_filter(high_pass_sobel, edited_image_canvas))
filters_menu.add_command(label="Limiarização (Thresholding)", command=lambda: apply_filter(thresholding_segmentation, edited_image_canvas))
filters_menu.add_command(label="Limiarização Adaptativa", command=lambda: apply_filter(otsu_segmentation, edited_image_canvas))
filters_menu.add_command(label="Erosion", command=lambda: apply_filter(erosion, edited_image_canvas))
filters_menu.add_command(label="Dilatation", command=lambda: apply_filter(dilatation, edited_image_canvas))
filters_menu.add_command(label="Open", command=lambda: apply_filter(open, edited_image_canvas))
filters_menu.add_command(label="Close", command=lambda: apply_filter(close, edited_image_canvas))

# Canvas para exibição de imagens
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()
