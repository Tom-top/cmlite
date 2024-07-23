import os

import numpy as np
import tifffile
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("Agg")


class TiffEditor:
    def __init__(self, master):

        # Expression data
        self.expression_matrix = None
        self.gene_list = None
        self.n_genes = None
        self.tissue_mask = None
        self.load_expression_matrix()
        self.load_transformed_cells()

        self.master = master
        master.title("MERFISH")
        self.file_path = None

        # Frame for buttons
        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side="top", fill="x", padx=5, pady=5)

        # Open button
        self.open_icon = ImageTk.PhotoImage(
            Image.open(r"/miscellaneous/open_icon.png")
            .resize((25, 25), Image.Resampling.LANCZOS))
        self.open_button = ttk.Button(self.button_frame, image=self.open_icon, command=self.open_tiff)
        self.open_button.pack(side="left", padx=2)

        # Save button
        self.save_icon = ImageTk.PhotoImage(
            Image.open(r"/miscellaneous/saving_icon.png")
            .resize((25, 25), Image.Resampling.LANCZOS))
        self.save_button = ttk.Button(self.button_frame, image=self.save_icon, command=self.save_mask)
        self.save_button.pack(side="left", padx=2)

        # Reset button
        self.reset_icon = ImageTk.PhotoImage(
            Image.open(r"/miscellaneous/reset_icon.png")
            .resize((25, 25), Image.Resampling.LANCZOS))
        self.reset_button = ttk.Button(self.button_frame, image=self.reset_icon, command=self.reset_mask)
        self.reset_button.pack(side="left", padx=2)

        # Generate plot button
        self.figure_canvas = None
        self.plot_button = ttk.Button(self.button_frame, text="Generate Plot", command=self.barplot)
        self.plot_button.pack(side="right", padx=2)

        # Tool selector combobox
        self.tool_var = tk.StringVar()
        self.tool_combobox = ttk.Combobox(self.button_frame, textvariable=self.tool_var,
                                          values=('Paint', 'Erase', "Magic_wand"))
        self.tool_combobox.set('Paint')  # set the default value
        self.tool_combobox.pack(side="left", padx=2)

        # Configure the main frame to hold the image display and the plot side by side
        self.main_frame = ttk.Frame(master)
        self.main_frame.pack(side="left", fill="both", expand=True)

        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.pack(side="left", fill="both", expand=True)

        # This frame will hold the plot
        self.plot_frame = ttk.Frame(self.main_frame, width=200)
        self.plot_frame.pack(side="right", fill="y", expand=False)

        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Label for the brush size adjustment scale
        self.brush_size = 1
        self.brush_size_var = tk.StringVar()
        self.brush_size_var.set(f'Brush Size: {self.brush_size}')  # Set initial value
        self.brush_size_label = ttk.Label(self.image_frame, textvariable=self.brush_size_var)
        self.brush_size_label.grid(row=1, column=1, sticky="ew")

        # Scrollbar for brush size adjustment
        self.scrollbar_h = ttk.Scale(self.image_frame, from_=1, to=10, orient="horizontal",
                                     command=self.update_brush_size)
        self.scrollbar_h.grid(row=1, column=0, sticky="ew")
        self.scrollbar_h.set(5)

        # Configure grid weights for resizing
        self.image_frame.columnconfigure(0, weight=1)
        self.image_frame.columnconfigure(1, weight=1)
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.rowconfigure(1, weight=1)

        # Initialize attributes for image and mask handling
        self.stack = None
        self.current_image = None
        self.mask_stack = []
        self.current_index = 0

        # Hover outline item
        self.hover_outline_id = None

        # Binds
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # For Windows and MacOS
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # For Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # For Linux scroll down

        self.zoom_level = 1
        self.zoomed_image = None
        self.canvas.bind("<Control-MouseWheel>", self.zoom_image)

    def hover(self, event):
        # Calculate the canvas position
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Calculate the image position (assuming no zoom for simplicity)
        img_x, img_y = int(canvas_x / 2), int(canvas_y / 2)

        if self.hover_outline_id is not None:
            # Remove the old hover outline
            self.canvas.delete(self.hover_outline_id)

        # Create a new hover outline
        # Calculate the bounding box for the oval
        x0, y0 = (img_x - self.brush_size) * 2, (img_y - self.brush_size) * 2
        x1, y1 = (img_x + self.brush_size) * 2, (img_y + self.brush_size) * 2
        self.hover_outline_id = self.canvas.create_oval(x0, y0, x1, y1, outline='green')

    def on_mousewheel(self, event):
        # For Windows and MacOS
        if event.num == 5 or event.delta == -120:
            self.scrollbar_v.set(self.scrollbar_v.get() + 1)
        if event.num == 4 or event.delta == 120:
            self.scrollbar_v.set(self.scrollbar_v.get() - 1)

        # Update the image index and display
        self.update_stack_index(self.scrollbar_v.get())

    def open_tiff(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            self.stack = [np.array(frame)[:, :184] for frame in ImageSequence.Iterator(Image.open(self.file_path))]
            self.outline = tifffile.imread(r"E:\tto\results\heatmaps\general\gubra_v6_atlas_outlines.tif")[:, :, :184]
            self.mask_stack = [np.zeros_like(self.stack[0], dtype=np.uint8) for _ in self.stack]
            self.current_index = 0
            self.display_image()

            # Adjust canvas size to the first image in the stack
            if self.stack:
                img_height, img_width = self.stack[0].shape[:2]
                self.canvas.config(width=img_width * 2, height=img_height * 2)  # Zoom factor of 2

                # Correct the range of the stack scale
                self.scrollbar_v = ttk.Scale(self.image_frame, from_=0, to=len(self.stack) - 1, orient="vertical",
                                             command=self.update_stack_index)
                self.scrollbar_v.grid(row=0, column=1, sticky="ns")  # This ensures it is vertical
                self.scrollbar_v.set((len(self.stack) - 1) / 2)
                self.canvas.bind("<Motion>", self.hover)
                # self.canvas.config(yscrollcommand=self.scrollbar_v.set)
                # self.scrollbar_v.config(command=self.canvas.yview)

    def update_stack_index(self, value):
        new_index = int(float(value))
        if new_index != self.current_index:
            self.current_index = new_index
            self.display_image()

    def update_brush_size(self, value):
        self.brush_size = int(float(value))
        self.brush_size_var.set(f'Brush Size: {self.brush_size}')  # Set initial value

    def display_image(self):
        original_image = self.stack[self.current_index]
        outline = self.outline[self.current_index]
        mask = self.mask_stack[self.current_index]

        colormap = matplotlib.colormaps.get_cmap('hot')
        colored_image = colormap(original_image)[:, :, :3]  # Get the RGB channels
        rgba_image = np.zeros((colored_image.shape[0], colored_image.shape[1], 4), dtype=np.uint8)
        rgba_image[..., :3] = (colored_image * 255).astype(np.uint8)  # Make sure to convert to uint8

        # Define the green color you want to use for painting
        target_color = np.array([0, 151, 255, 255], dtype=np.uint8)  # RGBA for green, as uint8
        # Define the red color for the outline
        outline_color = np.array([255, 255, 255, 25], dtype=np.uint8)  # RGBA for red, as uint8

        # Apply green where the mask is 255
        rgba_image[..., 3] = 255  # Set full opacity for RGB channels
        alpha = 0.5  # Transparency factor for the green paint
        rgba_image[mask == 255, :3] = rgba_image[mask == 255, :3] * (1 - alpha) + target_color[:3] * alpha
        rgba_image[mask == 255, 3] = target_color[3] * alpha  # Set the alpha for the green paint

        # Overlay the outline on the rgba_image
        rgba_image[outline == 255, :3] = outline_color[:3]  # Set the RGB channels to the outline color
        rgba_image[outline == 255, 3] = outline_color[3]  # Set the alpha channel for the outline

        # Convert to Image object and apply zoom
        image = Image.fromarray(rgba_image)
        zoomed_image = image.resize((image.width * 2, image.height * 2), Image.NEAREST)
        self.current_image = ImageTk.PhotoImage(zoomed_image)
        self.canvas.create_image(0, 0, image=self.current_image, anchor="nw")
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def paint(self, event):
        tool = self.tool_var.get()
        if tool == 'Paint':
            # Perform painting
            self.paint_pixels(event)
        elif tool == 'Erase':
            # Perform erasing
            self.erase_pixels(event)

        # Update hover outline
        self.update_hover_outline(event)

    def update_hover_outline(self, event):
        # This method should update the hover outline based on the current event position
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Assuming the zoom is 2x for both x and y axis
        img_x, img_y = int(canvas_x / 2), int(canvas_y / 2)

        if self.hover_outline_id is not None:
            self.canvas.delete(self.hover_outline_id)

        # Adjust the oval dimensions according to the brush size and zoom
        x0, y0 = (img_x - self.brush_size) * 2, (img_y - self.brush_size) * 2
        x1, y1 = (img_x + self.brush_size) * 2, (img_y + self.brush_size) * 2
        self.hover_outline_id = self.canvas.create_oval(x0, y0, x1, y1, outline='green')

    def paint_pixels(self, event):
        # Get canvas coordinates.
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Translate canvas coordinates to image coordinates, considering the zoom factor.
        img_x, img_y = int(canvas_x / 2), int(canvas_y / 2)

        # The center of the brush in 3D (z, y, x)
        center_coords = (self.current_index, img_y, img_x)

        # Get the affected points by the brush in 3D space.
        sphere_points = self.get_sphere_points(center=center_coords,
                                               radius=self.brush_size,
                                               shape=(len(self.stack),) + self.stack[0].shape)

        # Update the mask stack with the sphere points in 3D space.
        for z, y, x in zip(*sphere_points):
            if 0 <= z < len(self.stack) and 0 <= y < self.stack[0].shape[0] and 0 <= x < self.stack[0].shape[1]:
                self.mask_stack[z][y, x] = 255  # Paint with white on the mask.

        # Update the display with the new mask.
        self.display_image()

    def erase_pixels(self, event):
        # Logic for erasing pixels
        canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        img_x, img_y = int(canvas_x / 2), int(canvas_y / 2)
        center_coords = (self.current_index, img_y, img_x)

        # Here, rather than setting pixels to 255, we set them back to 0 to erase
        sphere_points = self.get_sphere_points(center=center_coords,
                                               radius=self.brush_size,
                                               shape=(len(self.stack),) + self.stack[0].shape)
        for z, y, x in zip(*sphere_points):
            if 0 <= z < len(self.stack):
                self.mask_stack[z][y, x] = 0  # Set mask pixel value to 0 to erase

        self.display_image()

    def get_sphere_points(self, center, radius, shape):
        # Generate points within a sphere for brush painting
        z, y, x = np.ogrid[-center[0]:shape[0] - center[0],
                  -center[1]:shape[1] - center[1],
                  -center[2]:shape[2] - center[2]]
        mask = x * x + y * y + z * z <= radius * radius
        return np.where(mask)

    def zoom_image(self, event):
        # Calculate zoom factor (might need adjustment for sensitivity)
        zoom_factor = 1.1 if event.delta > 0 else 0.9
        self.zoom_level *= zoom_factor

        # Update the display with the new zoom level
        self.update_zoomed_display(event.x, event.y, zoom_factor)

    def update_zoomed_display(self, center_x, center_y, zoom_factor):
        # Make sure we have an image to zoom
        if self.current_image is None:
            return

        # Calculate new size based on zoom factor
        new_width = int(self.current_image.width() * self.zoom_level)
        new_height = int(self.current_image.height() * self.zoom_level)

        # Resize image and update canvas item
        self.zoomed_image = self.current_image._PhotoImage__photo.zoom(int(self.zoom_level))
        self.canvas.itemconfig(self.canvas_image_id, image=self.zoomed_image)

        # Adjust the view so that we zoom towards the center_x, center_y
        self.canvas.scan_dragto(center_x - center_x * zoom_factor, center_y - center_y * zoom_factor, gain=1)

    def update_plot(self, event):
        selected_plot = self.plot_type_combobox.get()
        if selected_plot == "Barplot":
            self.plot_type_1()
        elif selected_plot == "Heatmaps":
            self.plot_type_2()
        else:
            print("Unknown plot type selected")

    def barplot(self):
        # Generate combobox to switch plots
        self.plot_var = tk.StringVar()
        self.plot_type_combobox = ttk.Combobox(self.button_frame, textvariable=self.plot_var,
                                               values=("Barplot", "Heatmaps"))
        self.plot_type_combobox.set('Barplot')  # set the default value
        self.plot_type_combobox.bind("<<ComboboxSelected>>", self.update_plot)
        self.plot_type_combobox.pack(side="right", padx=2)

        if self.figure_canvas:  # If a plot already exists, remove it
            self.figure_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots()

        self.avg_norm_gene_expression, self.hm_mid_planes = self.calculate_plot_data()

        self.idx_sorted_gene_expression = np.argsort(self.avg_norm_gene_expression)
        self.sorted_gene_expression = self.avg_norm_gene_expression[self.idx_sorted_gene_expression]
        self.sorted_gene_list = self.gene_list[self.idx_sorted_gene_expression]

        # Example plotting code (update with your actual plotting logic)
        ax.bar(np.arange(len(self.sorted_gene_expression[-25:])), self.sorted_gene_expression[-25:][::-1])
        ax.set_xticks(np.arange(len(self.sorted_gene_expression[-25:])))
        ax.set_xticklabels(self.sorted_gene_list[-25:][::-1], fontsize=8, rotation=45)
        ax.set_ylabel("normalized mean Log2(CPM+1) in region mask")
        ax.set_title("Normalized expression levels")
        plt.tight_layout()

        # Creating the matplotlib figure and embedding it in the Tkinter window
        self.figure_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.figure_canvas.draw()

        # Placing the canvas on the Tkinter window
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Function for plotting type 1
    def plot_type_1(self):
        if self.figure_canvas:  # If a plot already exists, remove it
            self.figure_canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots()

        # Example plotting code (update with your actual plotting logic)
        ax.bar(np.arange(len(self.sorted_gene_expression[-25:])), self.sorted_gene_expression[-25:][::-1])
        ax.set_xticks(np.arange(len(self.sorted_gene_expression[-25:])))
        ax.set_xticklabels(self.sorted_gene_list[-25:][::-1], fontsize=8, rotation=45)
        ax.set_ylabel("normalized mean Log2(CPM+1) in region mask")
        ax.set_title("Normalized expression levels")
        plt.tight_layout()

        # Creating the matplotlib figure and embedding it in the Tkinter window
        self.figure_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.figure_canvas.draw()

        # Placing the canvas on the Tkinter window
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Function for plotting type 2
    def plot_type_2(self):
        if self.figure_canvas:  # If a plot already exists, remove it
            self.figure_canvas.get_tk_widget().destroy()

        fig = plt.figure()

        sorted_hm_mid_planes = self.hm_mid_planes[self.idx_sorted_gene_expression]
        top_hm_mid_planes = sorted_hm_mid_planes[-25:][::-1]
        for i, (mp, nm) in enumerate(zip(top_hm_mid_planes, self.sorted_gene_list[-25:][::-1])):
            ax = plt.subplot(5, 5, i + 1)
            ax.imshow(mp, cmap="hot")
            ax.set_ylabel(f"{nm}")
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()

        # Creating the matplotlib figure and embedding it in the Tkinter window
        self.figure_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.figure_canvas.draw()

        # Placing the canvas on the Tkinter window
        self.figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_expression_matrix(self):
        heatmap_directory = r"E:\tto\results\heatmaps\Zhuang-ABCA-1_old"
        self.expression_matrix = tifffile.imread(os.path.join(heatmap_directory, "expression_matrix_old.tif"))
        self.gene_list = np.load(os.path.join(heatmap_directory, "gene_list_old.npy"))
        self.n_genes = len(self.gene_list)
        atlas_path = r"/gubra_and_ccfv3_alignment/gubra_template_coronal.tif"
        atlas = tifffile.imread(atlas_path)
        atlas_half = atlas[:, :, :184]
        self.tissue_mask = atlas_half > 0

    def load_transformed_cells(self):
        self.colors = None
        self.transformed_coordinates = None

    def calculate_plot_data(self):
        mask_stack = np.array(self.mask_stack)
        mask_bin = mask_stack == 255
        mask_bin_half = mask_bin[:, :, :184]
        mask_bin_half_sag = np.max(np.max(mask_bin_half, 1), 1)
        s, e = np.where(np.diff(mask_bin_half_sag) == 1)[0]
        mid = int((e + s) / 2)

        avg_gene_expression = []  # Average expression in the mask for each gene (uncorrected value)
        hm_mid_planes = []
        avg_norm_gene_expression = []  # Average normalized expression in the mask for each gene (normalized mean mask)

        for n, (gene, hm) in enumerate(zip(self.gene_list, self.expression_matrix)):
            print(f"Computing average signal for: {gene} {n + 1}/{self.n_genes}")
            avg_gene_exp_mask = np.mean(hm[mask_bin_half])
            avg_gene_expression.append(avg_gene_exp_mask)

            hm_mid_plane = hm[mid, :, :]
            hm_mid_planes.append(hm_mid_plane)

            tissue_not_mask = np.logical_and(self.tissue_mask, ~mask_bin_half)
            exp_brain = hm[tissue_not_mask]

            min_exp_brain, max_exp_brain = np.min(exp_brain), np.max(exp_brain)
            normalized_hm = (hm - min_exp_brain) / (max_exp_brain - min_exp_brain)

            avg_norm_gene_exp_mask = np.mean(normalized_hm[mask_bin_half])
            avg_norm_gene_expression.append(avg_norm_gene_exp_mask)

        hm_mid_planes = np.array(hm_mid_planes)
        avg_norm_gene_expression = np.array(avg_norm_gene_expression)

        return avg_norm_gene_expression, hm_mid_planes

    def calculate_plot_data_2(self):
        mask_stack = np.array(self.mask_stack)
        mask_bin = mask_stack == 255
        mask_bin_half = mask_bin[:, :, :184]

        # Use np.argmax to find the start and end of the true values along the sagittal axis
        mask_bin_half_sag = np.any(mask_bin_half, axis=(1, 2))
        s = np.argmax(mask_bin_half_sag)
        e = len(mask_bin_half_sag) - np.argmax(mask_bin_half_sag[::-1]) - 1
        mid = (e + s) // 2

        # Precompute the tissue_not_mask outside the loop
        tissue_not_mask = np.logical_and(self.tissue_mask, ~mask_bin_half)

        # Preallocate NumPy arrays for storage
        n_genes = len(self.gene_list)
        avg_gene_expression = np.empty(n_genes)
        hm_mid_planes = np.empty(
            (n_genes, mask_stack.shape[2], 184))  # Adjust the shape according to hm_mid_plane shape
        avg_norm_gene_expression = np.empty(n_genes)

        # Min and max expressions of the tissue_not_mask region for normalization
        min_exp_brain = np.min(self.expression_matrix[:, tissue_not_mask], axis=1)
        max_exp_brain = np.max(self.expression_matrix[:, tissue_not_mask], axis=1)

        for n, (gene, hm) in enumerate(zip(self.gene_list, self.expression_matrix)):
            print(f"Computing average signal for: {gene} {n + 1}/{n_genes}")

            # Mean expression in the mask for the current gene
            avg_gene_expression[n] = np.mean(hm[mask_bin_half])

            # Mid sagittal plane of expression matrix for the current gene
            hm_mid_planes[n] = hm[mid, :, :]

            # Normalize the expression matrix for the current gene
            # Use broadcasting to avoid creating large temporary arrays
            normalized_hm = (hm - min_exp_brain[n]) / (max_exp_brain[n] - min_exp_brain[n])

            # Mean normalized expression in the mask for the current gene
            avg_norm_gene_expression[n] = np.mean(normalized_hm[mask_bin_half])
        return avg_norm_gene_expression

    def reset_mask(self):
        # Check if there is a stack loaded
        if self.stack:
            # Reset all masks to zero
            self.mask_stack = [np.zeros_like(self.stack[0], dtype=np.uint8) for _ in self.stack]
            self.display_image()  # Redraw the current image without the mask

    def save_mask(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif")
        if file_path:
            tifffile.imwrite(file_path, self.mask_stack)
            # with tifffile.TiffWriter(file_path) as writer:
            # for mask in self.mask_stack:
            # writer.save(mask)


root = tk.Tk()
app = TiffEditor(root)
root.mainloop()
