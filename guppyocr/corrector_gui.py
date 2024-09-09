# Copyright (c) by Sebastian Kempf 2024

import argparse
from pathlib import Path

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from guppyocr.ocr_dataset import PlainDatasetReader, PlainDatasetWriter

import os

class SimpleGUI:
    def __init__(self, master, reader, writer):
        self.master = master
        self.master.title("GuppyOCR Official Correction Tool")

        # self.window_width = self.master.winfo_width()
        self.window_width = 1000
        self.font_size = 32
        self.font = ("Arial", self.font_size)

        self.reader = reader
        self.writer = writer

        # Initialize image path and text
        self.text_var = tk.StringVar()
        self.current_index = 0

        self.orig_image = None
        self.image_label = tk.Label(master)
        self.image_label.pack(side=tk.TOP)

        # Add an editable text field
        self.text_entry = tk.Entry(master, font=self.font, textvariable=self.text_var, width = 800)
        self.text_entry.pack()
        self.text_entry.bind("<Return>", lambda event: self.save())

        self.button_frame = tk.Frame(master)
        self.button_frame.pack(side=tk.BOTTOM)

        # Add a save button
        self.save_button = tk.Button(self.button_frame, text="Save", font=self.font, command=self.save)
        self.save_button.pack()

        self.master.bind("<Left>", lambda event: self.show_previous() if not self.text_entry.focus_get() else None)
        self.master.bind("<Right>", lambda event: self.show_next() if not self.text_entry.focus_get() else None)

        # Add back and forward buttons
        self.back_button = tk.Button(self.button_frame, text="←", font=self.font, command=self.show_previous)
        self.back_button.pack(side=tk.LEFT)

        self.curr = tk.IntVar(value=self.current_index)

        self.current = tk.Entry(self.button_frame, textvariable=self.curr, font=self.font)
        self.current.pack()
        # if go to index is pressed, load the image at that index
        self.current.bind("<Return>", lambda event: self.load())

        self.forward_button = tk.Button(self.button_frame, text="→", font=self.font,  command=self.show_next)
        self.forward_button.pack(side=tk.RIGHT)

        self.load()


    def load_image(self, image_path=None):
            image = Image.fromarray(image_path)
            self.orig_image = image
            aspect_ratio = image.width / image.height
            new_width = int(self.window_width - 2)  # Adjust padding
            new_height = max(1,int(new_width / aspect_ratio))
            image =  ImageTk.PhotoImage(image.resize((new_width, new_height)))
            self.image_label.configure(image=image)
            self.image_label.image = image

    def show_text(self, text):
        # Replace this with your own list of texts
        self.text_var.set(text)

    def show_previous(self):
        self.curr.set(self.curr.get()-1)
        self.load()
    
    def show_next(self):
        self.curr.set(self.curr.get()+1)
        self.load()
    
    def load(self):
        self.reader: PlainDatasetReader
        img, txt = self.reader.get_image(self.curr.get()), self.reader.get_text(self.curr.get())
        #_, img, txt = self.reader[self.curr.get()]
        self.show_text(txt)
        self.load_image(img)

    def save(self):
        self.writer.write(self.reader.files[self.curr.get()], self.orig_image, self.text_var.get())
        self.show_next()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="View images and associated text with possible editing"
    )
    parser.add_argument("--input_dir", type=str, help="Path to the input directory", default="/tmp/DiwPDFOut22/")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", default="/tmp/header_out")
    parser.add_argument(
        "--format",
        type=str,
        default="plain",
        help="Output format",
        choices=["parquet", "plain"],
    )
    args = parser.parse_args()

    if args.format == "parquet":
        pass
        # reader = ParquetDataset(Path(f"{args.input_dir}/dataset.parquet"))
        # writer = ParquetDatasetWriter(Path(f"{args.output_dir}/dataset.parquet"))
    else:
        reader = PlainDatasetReader(Path(args.input_dir))
        writer = PlainDatasetWriter(Path(args.output_dir), overwrite=True)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the main window
    root = tk.Tk()
    root.geometry("1000x600")
    root.resizable(True, True)

    # Instantiate the SimpleGUI class
    app = SimpleGUI(root, reader=reader, writer=writer)


    # Start the GUI event loop
    root.mainloop()
