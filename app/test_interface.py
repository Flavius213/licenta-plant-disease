from __future__ import annotations

import argparse
import sys
import tkinter as tk
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict_image import LoadedClassifier
from src.background_removal import leaf_on_checkerboard_image
from src.config import USER_FEEDBACK_DIR
from src.symbolic_rules import format_rule, rule_for_class
from src.train_classifier import BEST_MODEL_PATH


IMAGE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]


class PlantDiagnosisApp:
    def __init__(self, root: tk.Tk, checkpoint_path: Path):
        self.root = root
        self.checkpoint_path = checkpoint_path
        self.classifier: LoadedClassifier | None = None
        self.class_names: list[str] = []
        self.current_image_path: Path | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.remove_background_var = tk.BooleanVar(value=True)
        self.multi_crop_var = tk.BooleanVar(value=True)

        self.root.title("Plant Diagnosis - Test Interface")
        self.root.geometry("980x680")
        self.root.minsize(860, 600)

        self._configure_style()
        self._build_layout()
        self._load_model()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f7fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f5f7fb", foreground="#172033", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", foreground="#172033", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#f5f7fb", foreground="#172033", font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", background="#f5f7fb", foreground="#4b587c", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=18)
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.pack(fill=tk.X)

        title = ttk.Label(header, text="Plant Diagnosis", style="Title.TLabel")
        title.pack(side=tk.LEFT)

        self.status_label = ttk.Label(header, text="Loading model...", style="Status.TLabel")
        self.status_label.pack(side=tk.RIGHT)

        body = ttk.Frame(main)
        body.pack(fill=tk.BOTH, expand=True, pady=(18, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)

        controls = ttk.Frame(left_panel, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="ew")

        choose_button = ttk.Button(controls, text="Choose image", style="Accent.TButton", command=self.choose_image)
        choose_button.pack(side=tk.LEFT)

        diagnose_button = ttk.Button(controls, text="Diagnose", command=self.diagnose_current_image)
        diagnose_button.pack(side=tk.LEFT, padx=(8, 0))

        background_check = ttk.Checkbutton(
            controls,
            text="Remove background",
            variable=self.remove_background_var,
            command=self.refresh_current_image,
        )
        background_check.pack(side=tk.LEFT, padx=(8, 0))

        multi_crop_check = ttk.Checkbutton(
            controls,
            text="Multi-crop",
            variable=self.multi_crop_var,
            command=self.refresh_current_image,
        )
        multi_crop_check.pack(side=tk.LEFT, padx=(8, 0))

        self.class_choice = ttk.Combobox(controls, state="readonly", width=28)
        self.class_choice.pack(side=tk.LEFT, padx=(16, 0))

        feedback_button = ttk.Button(controls, text="Save feedback", command=self.save_feedback)
        feedback_button.pack(side=tk.LEFT, padx=(8, 0))

        self.image_path_label = ttk.Label(left_panel, text="No image selected", style="Panel.TLabel")
        self.image_path_label.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        self.preview_label = ttk.Label(left_panel, text="Image preview", anchor=tk.CENTER, style="Panel.TLabel")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(14, 0))

        right_panel = ttk.Frame(body, style="Panel.TFrame", padding=14)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(9, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        result_title = ttk.Label(right_panel, text="Result", style="Panel.TLabel", font=("Segoe UI", 13, "bold"))
        result_title.grid(row=0, column=0, sticky="w")

        self.result_text = tk.Text(
            right_panel,
            wrap=tk.WORD,
            height=24,
            borderwidth=0,
            padx=8,
            pady=8,
            font=("Segoe UI", 10),
            background="#ffffff",
            foreground="#172033",
        )
        self.result_text.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.result_text.insert(tk.END, "Choose a leaf image to run a diagnosis.")
        self.result_text.configure(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=(12, 0))
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def _load_model(self) -> None:
        try:
            self.classifier = LoadedClassifier(self.checkpoint_path)
            self.class_names = list(self.classifier.classes)
            self.class_choice.configure(values=self.class_names)
            if self.class_names:
                self.class_choice.set(self.class_names[0])
            self.status_label.configure(text=f"Model loaded: {self.checkpoint_path.name}")
        except Exception as exc:
            self.status_label.configure(text="The model could not be loaded")
            messagebox.showerror("Model error", str(exc))

    def choose_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose an image",
            initialdir=str(PROJECT_DIR),
            filetypes=IMAGE_TYPES,
        )
        if not path:
            return

        self.current_image_path = Path(path)
        self.image_path_label.configure(text=str(self.current_image_path))
        self._show_preview(self.current_image_path)
        self.diagnose_current_image()

    def refresh_current_image(self) -> None:
        if self.current_image_path is None:
            return
        self._show_preview(self.current_image_path)
        self.diagnose_current_image()

    def _show_preview(self, image_path: Path) -> None:
        try:
            if self.remove_background_var.get():
                image = leaf_on_checkerboard_image(image_path)
            else:
                with Image.open(image_path) as opened_image:
                    image = opened_image.convert("RGB")
            image.thumbnail((420, 420))
            self.preview_image = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_image, text="")
        except Exception as exc:
            self.preview_label.configure(image="", text="The image cannot be displayed")
            messagebox.showerror("Image error", str(exc))

    def diagnose_current_image(self) -> None:
        if self.classifier is None:
            messagebox.showwarning("Missing model", "The model is not loaded.")
            return
        if self.current_image_path is None:
            messagebox.showinfo("Missing image", "Choose an image first.")
            return

        self.status_label.configure(text="Running diagnosis...")
        self.root.update_idletasks()

        try:
            if self.multi_crop_var.get():
                result = self.classifier.predict_multi_crop(
                    self.current_image_path,
                    top_k=3,
                    remove_background=self.remove_background_var.get(),
                    max_crops=8,
                    detection_backend="opencv",
                )
                predictions = result.final_predictions
            else:
                result = None
                predictions = self.classifier.predict(
                    self.current_image_path,
                    top_k=3,
                    remove_background=self.remove_background_var.get(),
                )
            best_class, best_probability = predictions[0]
            text = format_rule(best_class, best_probability, rule_for_class(best_class))
            if result is not None:
                text += "\n\nMulti-crop analysis:\n"
                text += f"- generated crops: {len(result.crop_predictions)}\n"
                text += "- vote: confidence-weighted majority\n"
                text += "- relevant crops:\n"
                for crop in result.crop_predictions[:8]:
                    text += (
                        f"  crop {crop.crop_index} [{crop.source}] -> "
                        f"{crop.predicted_class} ({crop.confidence * 100:.2f}%, weight={crop.weight:.2f})\n"
                    )
            if len(predictions) > 1:
                text += "\n\nAlternative:\n"
                text += "\n".join(
                    f"- {class_name}: {probability * 100:.2f}%" for class_name, probability in predictions[1:]
                )
            self._set_result(text)
            self.status_label.configure(text="Diagnosis complete")
        except Exception as exc:
            self.status_label.configure(text="Diagnosis error")
            messagebox.showerror("Diagnosis error", str(exc))

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)

    def save_feedback(self) -> None:
        if self.current_image_path is None:
            messagebox.showinfo("Missing image", "Choose an image first.")
            return

        class_name = self.class_choice.get()
        if not class_name:
            messagebox.showinfo("Missing class", "Choose the correct class for the image.")
            return

        destination_dir = USER_FEEDBACK_DIR / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(destination_dir.iterdir())) + 1
        if self.remove_background_var.get():
            destination_path = destination_dir / f"{self.current_image_path.stem}_{index:04d}_leaf_bg.png"
            leaf_on_checkerboard_image(self.current_image_path).save(destination_path, "PNG", optimize=True)
        else:
            destination_path = destination_dir / f"{self.current_image_path.stem}_{index:04d}{self.current_image_path.suffix.lower()}"
            shutil.copy2(self.current_image_path, destination_path)
        self.status_label.configure(text=f"Feedback saved: {class_name}")
        messagebox.showinfo("Feedback saved", f"The image was saved for fine-tuning in class {class_name}.")


def find_sample_image() -> Path:
    test_dir = PROJECT_DIR / "data" / "final" / "test"
    for path in test_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            return path
    raise FileNotFoundError("No test images were found in data/final/test.")


def run_self_test(
    image_path: Path | None,
    checkpoint_path: Path,
    *,
    remove_background: bool,
    multi_crop: bool,
) -> None:
    selected_image = image_path or find_sample_image()
    classifier = LoadedClassifier(checkpoint_path, remove_background=remove_background)
    if multi_crop:
        result = classifier.predict_multi_crop(selected_image, top_k=3, max_crops=8)
        predictions = result.final_predictions
        print(f"[OK] Generated crops: {len(result.crop_predictions)}")
        for crop in result.crop_predictions:
            print(
                f"[OK] crop {crop.crop_index} [{crop.source}] -> "
                f"{crop.predicted_class} ({crop.confidence * 100:.2f}%, weight={crop.weight:.2f})"
            )
    else:
        predictions = classifier.predict(selected_image, top_k=3)
    best_class, best_probability = predictions[0]
    print(f"[OK] Image: {selected_image}")
    print(format_rule(best_class, best_probability, rule_for_class(best_class)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tkinter test interface for plant diagnosis.")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_PATH), help="Checkpoint model.")
    parser.add_argument("--self-test", action="store_true", help="Run a prediction without the graphical interface.")
    parser.add_argument("--image", help="Optional image for self-test.")
    parser.add_argument("--remove-background", action="store_true", help="Remove the background during self-test.")
    parser.add_argument("--single-crop", action="store_true", help="Disable multi-crop during self-test.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    image_path = Path(args.image) if args.image else None

    if args.self_test:
        run_self_test(
            image_path,
            checkpoint_path,
            remove_background=args.remove_background,
            multi_crop=not args.single_crop,
        )
        return

    root = tk.Tk()
    PlantDiagnosisApp(root, checkpoint_path)
    root.mainloop()


if __name__ == "__main__":
    main()
