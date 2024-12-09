import tkinter as tk
from image_handler import load_image, display_image
from filters import *

img_cv = None
processed_img_cv = None
slider_window_ref = None  


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


def slider_window(root, canvas, filter_function, label, param_name, param_range, default_value):
    global slider_window_ref

    # Fecha a janela anterior, se existir
    if slider_window_ref is not None:
        slider_window_ref.destroy()

    # Cria uma nova janela do slider
    slider_window_ref = tk.Toplevel(root)
    slider_window_ref.title(f"{label} Slider")

    def on_slider_change(value):
        nonlocal default_value
        default_value = int(value)  # Atualiza o valor padrão com o slider
        apply_filter(filter_function, canvas, **{param_name: default_value})

    slider = tk.Scale(
        slider_window_ref,
        from_=param_range[0],
        to=param_range[1],
        orient=tk.HORIZONTAL,
        label=label,
        command=on_slider_change,
    )
    slider.pack(padx=20, pady=20)
    slider.set(default_value)


# Configuração da janela principal
root = tk.Tk()
root.title("Image Processing App")
root.geometry("1085x550")
root.config(bg="#2e2e2e")

# Barra de menus
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Image", command=lambda: set_img_cv(load_image(original_image_canvas, edited_image_canvas, slider_window_ref)))
file_menu.add_command(label="Exit", command=root.quit)

filters_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Filters", menu=filters_menu)

# Adiciona os filtros ao menu
filters_menu.add_command(label="Low Pass Filter", command=lambda: apply_filter(low_pass, edited_image_canvas))
filters_menu.add_command(label="Low Pass Filter Gaussian", command=lambda: slider_window(root, edited_image_canvas, low_pass_gaussian, "Sigma", "sigma", (1, 10), 3))
filters_menu.add_command(label="Low Pass Filter Media", command=lambda: apply_filter(low_pass_media, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter", command=lambda: apply_filter(high_pass, edited_image_canvas))
filters_menu.add_command(label="High Pass Filter Laplacian", command=lambda: slider_window(root, edited_image_canvas, high_pass_laplacian, "Kernel Size", "kernel_value", (1, 15), 3))
filters_menu.add_command(label="High Pass Filter Sobel", command=lambda: apply_filter(high_pass_sobel, edited_image_canvas))
filters_menu.add_command(label="Limiarização (Thresholding)", command=lambda: apply_filter(thresholding_segmentation, edited_image_canvas))
filters_menu.add_command(label="Limiarização Adaptativa", command=lambda: apply_filter(otsu_segmentation, edited_image_canvas))
filters_menu.add_command(label="Erosion", command=lambda: slider_window(root, edited_image_canvas, erosion, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Dilatation", command=lambda: slider_window(root, edited_image_canvas, dilatation, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Open", command=lambda: slider_window(root, edited_image_canvas, open, "Kernel Size", "kernel_size", (1, 15), 5))
filters_menu.add_command(label="Close", command=lambda: slider_window(root, edited_image_canvas, close, "Kernel Size", "kernel_size", (1, 15), 5))

# Canvas para exibição de imagens
original_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
original_image_canvas.grid(row=0, column=0, padx=20, pady=20)

edited_image_canvas = tk.Canvas(root, width=500, height=500, bg="#2e2e2e", highlightthickness=1, highlightbackground="white")
edited_image_canvas.grid(row=0, column=1, padx=20, pady=20)

root.mainloop()