import os
import sys
import csv
import argparse
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

# Local import (stub for now; Ray will plug in real models)
import predict_api

APP_TITLE = "Fossil Classifier (v0.3.3 — safe mode)"
MIN_WIN = (980, 600)
SUPPORTED_MODELS = ["VGG16", "ResNet18", "CustomCNN"]
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".heic")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.minsize(*MIN_WIN)
        self._image_path = None
        self._last_result = None  # for single save
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # Sidebar
        sidebar = ttk.Frame(self, padding=12)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsw")

        ttk.Label(sidebar, text="Actions", font=("Helvetica", 12, "bold")).grid(sticky="w", pady=(0, 8))
        ttk.Button(sidebar, text="Choose Image (macOS)", command=self._choose_macos_file).grid(sticky="ew")
        ttk.Button(sidebar, text="Paste Path", command=self._on_paste_path).grid(sticky="ew", pady=4)
        self.save_btn = ttk.Button(sidebar, text="Save to CSV", command=self._on_save_csv, state="disabled")
        self.save_btn.grid(sticky="ew")
        # NEW buttons
        ttk.Button(sidebar, text="Save Console Log", command=self._save_log).grid(sticky="ew")
        ttk.Button(sidebar, text="Open outputs folder", command=self._open_outputs).grid(sticky="ew")
        ttk.Button(sidebar, text="Clear Console", command=self._clear_console).grid(sticky="ew")

        ttk.Separator(sidebar).grid(sticky="ew", pady=10)
        ttk.Label(sidebar, text="Batch", font=("Helvetica", 11, "bold")).grid(sticky="w")
        ttk.Label(sidebar, text="Model for batch:").grid(sticky="w", pady=(4, 2))
        self.batch_model = tk.StringVar(value=SUPPORTED_MODELS[0])
        ttk.Combobox(sidebar, textvariable=self.batch_model, values=SUPPORTED_MODELS, state="readonly").grid(sticky="ew")
        self.recurse = tk.BooleanVar(value=False)  # include subfolders
        ttk.Checkbutton(sidebar, text="Include subfolders", variable=self.recurse).grid(sticky="w", pady=(6, 0))
        ttk.Button(sidebar, text="Choose Folder → CSV (macOS)", command=self._choose_macos_folder).grid(sticky="ew", pady=(6, 2))
        ttk.Button(sidebar, text="Paste Folder Path → CSV", command=self._on_paste_folder).grid(sticky="ew")

        ttk.Label(sidebar, text="\nModels", font=("Helvetica", 10, "bold")).grid(sticky="w")
        for name in SUPPORTED_MODELS:
            ttk.Button(sidebar, text=name, command=lambda n=name: self._on_predict(n)).grid(sticky="ew", pady=2)

        # Main area (Preview placeholder + Console)
        main = ttk.Frame(self, padding=12)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(main, text="No image loaded (safe mode)", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Console
        console_frame = ttk.Frame(self, padding=(12, 0, 12, 12))
        console_frame.grid(row=1, column=1, sticky="nsew")
        console_frame.grid_columnconfigure(0, weight=1)
        console_frame.grid_rowconfigure(0, weight=1)

        ttk.Label(console_frame, text="Console", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
        self.console = tk.Text(console_frame, height=12, wrap="word")
        self.console.grid(row=1, column=0, sticky="nsew")
        ttk.Scrollbar(console_frame, command=self.console.yview).grid(row=1, column=1, sticky="ns")
        self.console.configure(yscrollcommand=lambda *args: None)

        # Status bar
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", padding=(12, 6)).grid(row=1, column=0, columnspan=2, sticky="ew")

    # ---------- Helpers ----------
    def _log(self, msg: str):
        self.console.insert("end", msg + "\n")
        self.console.see("end")

    def _clear_console(self):
        self.console.delete("1.0", "end")

    def _open_outputs(self):
        os.makedirs("outputs", exist_ok=True)
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", "outputs"])
            elif sys.platform.startswith("win"):
                os.startfile(os.path.abspath("outputs"))  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", "outputs"])
            self.status.set("Opened outputs/")
        except Exception as e:
            self._log(f"Open outputs error: {e}")
            messagebox.showerror("Open outputs error", str(e))

    def _write_csv_row(self, row: dict):
        os.makedirs("outputs", exist_ok=True)
        csv_path = os.path.join("outputs", "predictions.csv")
        fields = ["timestamp", "filename", "image_path", "model", "top_class", "confidence"]
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)
        return csv_path

    # ---------- Single-file actions ----------
    def _choose_macos_file(self):
        try:
            script = 'POSIX path of (choose file of type {"public.png","public.jpeg","public.tiff","public.heic","public.bmp","public.gif"} with prompt "Select image for Fossil Classifier")'
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            path = (out.stdout or "").strip()
            if path and os.path.isfile(path):
                self._load_path(path)
        except Exception as e:
            self._log(f"Chooser error: {e}")
            messagebox.showerror("Chooser error", str(e))

    def _on_paste_path(self):
        path = simpledialog.askstring("Paste Path", "Paste full image path:")
        if not path:
            return
        path = path.strip().strip('"')  # allow quoted paths
        if not os.path.isfile(path):
            messagebox.showwarning("Path not found", f"No file at:\n{path}")
            return
        self._load_path(path)

    def _load_path(self, path: str):
        self._image_path = path
        self.preview_label.configure(text=f"Loaded: {os.path.basename(path)}\n(safe-mode preview disabled)")
        self.status.set(f"Loaded: {os.path.basename(path)}")
        self._log(f"Image selected: {path}")
        self.save_btn.config(state="disabled")
        self._last_result = None

    def _on_predict(self, model_name: str):
        if not self._image_path:
            messagebox.showinfo("Predict", "Please load an image first (Paste Path or Choose Image).")
            return
        self._log(f"Running prediction with {model_name}…")
        self.status.set(f"Predicting with {model_name}…")
        try:
            result = predict_api.predict(self._image_path, model_name)
            pretty = (
                f"Model: {result.get('model')}\n"
                f"Top Class: {result.get('top_class')}\n"
                f"Confidence: {result.get('confidence'):.3f}\n"
            )
            self._log(pretty)
            self.status.set("Done")
            # Prepare for CSV export
            self._last_result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "filename": os.path.basename(self._image_path),
                "image_path": self._image_path,
                "model": result.get("model"),
                "top_class": result.get("top_class"),
                "confidence": float(result.get("confidence", 0.0)),
            }
            self.save_btn.config(state="normal")
        except Exception as e:
            self._log(f"ERROR: {e}")
            messagebox.showerror("Prediction failed", str(e))
            self.status.set("Error")

    def _on_save_csv(self):
        if not self._last_result:
            messagebox.showinfo("Save to CSV", "Run a prediction first.")
            return
        csv_path = self._write_csv_row(self._last_result)
        self._log(f"Saved row to {csv_path}")
        self.status.set(f"Saved → {csv_path}")

    # Save console contents to a text file
    def _save_log(self):
        os.makedirs("outputs", exist_ok=True)
        path = os.path.join("outputs", f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.console.get("1.0", "end-1c"))
            self._log(f"Saved console to {path}")
            self.status.set(f"Saved → {path}")
        except Exception as e:
            self._log(f"Console save error: {e}")
            messagebox.showerror("Console Save Error", str(e))

    # ---------- Batch actions ----------
    def _choose_macos_folder(self):
        try:
            script = 'POSIX path of (choose folder with prompt "Select folder of images for Fossil Classifier")'
            out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            folder = (out.stdout or "").strip()
            if folder and os.path.isdir(folder):
                self._run_batch(folder)
        except Exception as e:
            self._log(f"Folder chooser error: {e}")
            messagebox.showerror("Chooser error", str(e))

    def _on_paste_folder(self):
        folder = simpledialog.askstring("Paste Folder Path", "Paste full folder path:")
        if not folder:
            return
        folder = folder.strip().strip('"')
        if not os.path.isdir(folder):
            messagebox.showwarning("Folder not found", f"No folder at:\n{folder}")
            return
        self._run_batch(folder)

    def _run_batch(self, folder: str):
        model = self.batch_model.get()
        self._log(f"Batch start → folder: {folder} | model: {model}")

        # Collect images (optionally recurse into subfolders)
        if self.recurse.get():
            image_paths = []
            for root, _, files in os.walk(folder):
                for f in files:
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                        image_paths.append(os.path.join(root, f))
        else:
            image_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
            ]

        image_paths.sort()
        if not image_paths:
            self._log("No supported images found in folder.")
            messagebox.showinfo("Batch", "No supported images in that folder.")
            return

        ok, err = 0, 0
        csv_path = None
        for i, path in enumerate(image_paths, 1):
            try:
                result = predict_api.predict(path, model)
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "filename": os.path.basename(path),
                    "image_path": path,
                    "model": result.get("model"),
                    "top_class": result.get("top_class"),
                    "confidence": float(result.get("confidence", 0.0)),
                }
                csv_path = self._write_csv_row(row)
                ok += 1
                self._log(f"[{i}/{len(image_paths)}] Saved → {os.path.basename(path)}")
            except Exception as e:
                err += 1
                self._log(f"[{i}/{len(image_paths)}] ERROR {os.path.basename(path)}: {e}")
                continue

        if csv_path:
            self._log(f"Batch done: {ok} saved, {err} errors. CSV: {csv_path}")
        else:
            self._log(f"Batch done: {ok} saved, {err} errors.")
        self.status.set(f"Batch done → {ok} saved, {err} errors")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Optional image path to load on start")
    args = parser.parse_args()

    app = App()
    if args.image and os.path.isfile(args.image):
        app._load_path(args.image)
    app.mainloop()

if __name__ == "__main__":
    main()
