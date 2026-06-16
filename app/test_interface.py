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


COLORS = {
    "page": "#eef3ef",
    "card": "#ffffff",
    "card_border": "#d8e4dc",
    "ink": "#17251f",
    "muted": "#66766f",
    "accent": "#1b6b5c",
    "accent_dark": "#17443b",
    "accent_soft": "#e4f3ed",
    "danger": "#9f2f2f",
    "warning": "#8a5a15",
}

STATUS_COLORS = {
    "loading": ("#e7eefb", "#27405f"),
    "busy": ("#fff3d8", COLORS["warning"]),
    "ok": ("#dff4ea", COLORS["accent_dark"]),
    "error": ("#fde5e2", COLORS["danger"]),
    "neutral": ("#e9eeeb", COLORS["muted"]),
}


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
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)
        self.root.configure(background=COLORS["page"])

        self._configure_style()
        self._build_layout()
        self._load_model()

    def _configure_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["page"])
        style.configure("Panel.TFrame", background=COLORS["card"], relief="solid", borderwidth=1)
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure("TFrame", background=COLORS["page"])
        style.configure("TLabel", background=COLORS["page"], foreground=COLORS["ink"], font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["ink"], font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=COLORS["page"], foreground=COLORS["ink"], font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["page"], foreground=COLORS["muted"], font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=COLORS["card"], foreground=COLORS["ink"], font=("Segoe UI", 13, "bold"))
        style.configure("Small.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Path.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("Soft.TButton", font=("Segoe UI", 10), padding=(12, 8))
        style.configure("Modern.TCheckbutton", background=COLORS["card"], foreground=COLORS["ink"], font=("Segoe UI", 10))
        style.map(
            "Modern.TCheckbutton",
            background=[("active", COLORS["card"])],
            foreground=[("disabled", "#9aa8a1")],
        )
        style.configure(
            "TCombobox",
            fieldbackground="#f8fbf9",
            background="#f8fbf9",
            foreground=COLORS["ink"],
            padding=(6, 5),
        )

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=24, style="App.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main, style="App.TFrame")
        header.pack(fill=tk.X)

        title_block = ttk.Frame(header, style="App.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.X, expand=True)

        title = ttk.Label(title_block, text="Plant Diagnosis", style="Title.TLabel")
        title.pack(anchor="w")

        subtitle = ttk.Label(
            title_block,
            text="Desktop test workspace for image diagnosis, crop voting and feedback capture",
            style="Subtitle.TLabel",
        )
        subtitle.pack(anchor="w", pady=(3, 0))

        self.status_label = tk.Label(
            header,
            text="",
            padx=14,
            pady=7,
            bd=0,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_label.pack(side=tk.RIGHT, padx=(18, 0))
        self._set_status("Loading model...", "loading")

        body = ttk.Frame(main, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True, pady=(22, 0))
        body.columnconfigure(0, weight=3, uniform="content")
        body.columnconfigure(1, weight=2, uniform="content")
        body.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left_panel.rowconfigure(2, weight=1)
        left_panel.columnconfigure(0, weight=1)

        image_header = ttk.Frame(left_panel, style="Card.TFrame")
        image_header.grid(row=0, column=0, sticky="ew")
        image_header.columnconfigure(0, weight=1)

        image_title = ttk.Label(image_header, text="Image input", style="Section.TLabel")
        image_title.grid(row=0, column=0, sticky="w")

        self.image_path_label = ttk.Label(
            image_header,
            text="No image selected",
            style="Path.TLabel",
            wraplength=560,
        )
        self.image_path_label.grid(row=1, column=0, sticky="ew", pady=(5, 0))

        controls = ttk.Frame(image_header, style="Card.TFrame")
        controls.grid(row=0, column=1, rowspan=2, sticky="e", padx=(18, 0))

        choose_button = ttk.Button(controls, text="Choose image", style="Accent.TButton", command=self.choose_image)
        choose_button.pack(side=tk.LEFT, padx=(0, 8))

        diagnose_button = ttk.Button(controls, text="Diagnose", style="Soft.TButton", command=self.diagnose_current_image)
        diagnose_button.pack(side=tk.LEFT)

        self.preview_frame = tk.Frame(
            left_panel,
            background="#f8fbf9",
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
        )
        self.preview_frame.grid(row=2, column=0, sticky="nsew", pady=(16, 0))
        self.preview_frame.rowconfigure(0, weight=1)
        self.preview_frame.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            self.preview_frame,
            text="Image preview\n\nChoose a leaf photo to start",
            anchor=tk.CENTER,
            justify=tk.CENTER,
            background="#f8fbf9",
            foreground=COLORS["muted"],
            font=("Segoe UI", 12, "bold"),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        settings = ttk.Frame(left_panel, style="Card.TFrame")
        settings.grid(row=3, column=0, sticky="ew", pady=(16, 0))
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(2, weight=1)

        settings_title = ttk.Label(settings, text="Analysis settings", style="Section.TLabel")
        settings_title.grid(row=0, column=0, sticky="w", columnspan=3)

        background_check = ttk.Checkbutton(
            settings,
            text="Remove background",
            variable=self.remove_background_var,
            command=self.refresh_current_image,
            style="Modern.TCheckbutton",
        )
        background_check.grid(row=1, column=0, sticky="w", pady=(12, 0))

        multi_crop_check = ttk.Checkbutton(
            settings,
            text="Multi-crop voting",
            variable=self.multi_crop_var,
            command=self.refresh_current_image,
            style="Modern.TCheckbutton",
        )
        multi_crop_check.grid(row=1, column=1, sticky="w", pady=(12, 0), padx=(10, 0))

        class_block = ttk.Frame(settings, style="Card.TFrame")
        class_block.grid(row=1, column=2, sticky="e", pady=(12, 0))
        class_label = ttk.Label(class_block, text="Feedback class", style="Small.TLabel")
        class_label.pack(side=tk.LEFT, padx=(0, 8))
        self.class_choice = ttk.Combobox(class_block, state="readonly", width=25)
        self.class_choice.pack(side=tk.LEFT)

        feedback_button = ttk.Button(settings, text="Save feedback", style="Soft.TButton", command=self.save_feedback)
        feedback_button.grid(row=2, column=2, sticky="e", pady=(12, 0))

        right_panel = ttk.Frame(body, style="Panel.TFrame", padding=18)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right_panel.rowconfigure(1, weight=1)
        right_panel.columnconfigure(0, weight=1)

        result_header = ttk.Frame(right_panel, style="Card.TFrame")
        result_header.grid(row=0, column=0, sticky="ew")
        result_header.columnconfigure(0, weight=1)

        result_title = ttk.Label(result_header, text="Diagnosis result", style="Section.TLabel")
        result_title.grid(row=0, column=0, sticky="w")
        result_hint = ttk.Label(result_header, text="Top prediction, rules and crop votes", style="Small.TLabel")
        result_hint.grid(row=1, column=0, sticky="w", pady=(4, 0))

        text_frame = tk.Frame(
            right_panel,
            background="#f8fbf9",
            highlightbackground=COLORS["card_border"],
            highlightthickness=1,
        )
        text_frame.grid(row=1, column=0, sticky="nsew", pady=(16, 0))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self.result_text = tk.Text(
            text_frame,
            wrap=tk.WORD,
            height=24,
            borderwidth=0,
            padx=14,
            pady=14,
            font=("Segoe UI", 10),
            background="#f8fbf9",
            foreground=COLORS["ink"],
            insertbackground=COLORS["accent"],
            selectbackground="#cae7dc",
        )
        self.result_text.grid(row=0, column=0, sticky="nsew")
        self.result_text.tag_configure("heading", foreground=COLORS["accent_dark"], font=("Segoe UI", 10, "bold"))
        self.result_text.tag_configure("muted", foreground=COLORS["muted"])
        self.result_text.insert(tk.END, "Choose a leaf image to run a diagnosis.")
        self.result_text.configure(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_text.configure(yscrollcommand=scrollbar.set)

    def _set_status(self, text: str, state: str = "neutral") -> None:
        background, foreground = STATUS_COLORS.get(state, STATUS_COLORS["neutral"])
        self.status_label.configure(text=text, background=background, foreground=foreground)

    def _load_model(self) -> None:
        try:
            self.classifier = LoadedClassifier(self.checkpoint_path)
            self.class_names = list(self.classifier.classes)
            self.class_choice.configure(values=self.class_names)
            if self.class_names:
                self.class_choice.set(self.class_names[0])
            self._set_status(f"Model loaded: {self.checkpoint_path.name}", "ok")
        except Exception as exc:
            self._set_status("The model could not be loaded", "error")
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

        self._set_status("Running diagnosis...", "busy")
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
            self._set_status("Diagnosis complete", "ok")
        except Exception as exc:
            self._set_status("Diagnosis error", "error")
            messagebox.showerror("Diagnosis error", str(exc))

    def _set_result(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        for line in text.splitlines(keepends=True):
            start = self.result_text.index(tk.INSERT)
            self.result_text.insert(tk.END, line)
            clean_line = line.strip()
            end = self.result_text.index(tk.INSERT)
            if clean_line.endswith(":") or clean_line.startswith("Prediction:"):
                self.result_text.tag_add("heading", start, end)
            elif clean_line.startswith("-") or clean_line.startswith("crop"):
                self.result_text.tag_add("muted", start, end)
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
        self._set_status(f"Feedback saved: {class_name}", "ok")
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
