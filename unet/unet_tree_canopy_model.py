import sys
import os
import yaml
import argparse

# ---------------------------------------------------------------------------
# Shared configuration helpers (used by both GUI and headless modes)
# ---------------------------------------------------------------------------

# Resolve project root: one level up from the unet/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DEFAULT_CONFIG = {
    "train_images": os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "train", "images"),
    "train_labels": os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "train", "labels"),
    "val_images":   os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "val",   "images"),
    "val_labels":   os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "val",   "labels"),
    "test_images":  os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "test",  "images"),
    "test_labels":  os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "test",  "labels"),
    "predict_images": os.path.join(PROJECT_ROOT, "BH_CT_Data", "unet_dataset", "test", "images"),
    "output_tif": os.path.join(PROJECT_ROOT, "unet", "predict_output.tif"),
    "model_path": "",
    "use_aug": False,
    "threshold": 0.5,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 10,
}


def resolve_path(path_str: str) -> str:
    """Resolve a path relative to the project root, unless it is already absolute."""
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(PROJECT_ROOT, path_str))


def load_or_create_config(config_path: str) -> dict:
    """Load config from *config_path*, filling missing keys from DEFAULT_CONFIG."""
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(config_path):
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
        # Merge: loaded values override defaults, resolve relative paths
        for key, val in loaded.items():
            if key in cfg:
                cfg[key] = val
    # Resolve all path-like values
    path_keys = [
        "train_images", "train_labels", "val_images", "val_labels",
        "test_images", "test_labels", "predict_images", "output_tif", "model_path",
    ]
    for k in path_keys:
        cfg[k] = resolve_path(str(cfg.get(k, "")))
    return cfg


def save_config(cfg: dict, config_path: str):
    """Save config dict to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


# ---------------------------------------------------------------------------
# ================================  HEADLESS (CLI) MODE  ====================
# ---------------------------------------------------------------------------

def print_banner():
    print("=" * 60)
    print("  U-Net Segmentation Toolbox  —  Headless (CLI) Mode")
    print("=" * 60)


def print_config(cfg: dict):
    """Pretty-print the current configuration."""
    print("\n--- Current Configuration ---")
    print(f"  Train Images : {cfg['train_images']}")
    print(f"  Train Labels : {cfg['train_labels']}")
    print(f"  Val Images   : {cfg['val_images']}")
    print(f"  Val Labels   : {cfg['val_labels']}")
    print(f"  Test Images  : {cfg['test_images']}")
    print(f"  Test Labels  : {cfg['test_labels']}")
    print(f"  Predict Imgs : {cfg['predict_images']}")
    print(f"  Output TIF   : {cfg['output_tif']}")
    print(f"  Model Path   : {cfg['model_path']}")
    print(f"  Learning Rate: {cfg['learning_rate']}")
    print(f"  Batch Size   : {cfg['batch_size']}")
    print(f"  Epochs       : {cfg['epochs']}")
    print(f"  Threshold    : {cfg['threshold']}")
    print(f"  Use Aug      : {cfg['use_aug']}")
    print("-----------------------------\n")


def ask_modify_config(cfg: dict) -> dict:
    """Interactively ask the user if they want to modify any config value."""
    editable = [
        ("train_images",    "Train Images dir"),
        ("train_labels",    "Train Labels dir"),
        ("val_images",      "Validation Images dir"),
        ("val_labels",      "Validation Labels dir"),
        ("test_images",     "Test Images dir"),
        ("test_labels",     "Test Labels dir"),
        ("predict_images",  "Predict Images dir"),
        ("output_tif",      "Output TIF path"),
        ("model_path",      "Model weight path (.pt)"),
        ("learning_rate",   "Learning Rate"),
        ("batch_size",      "Batch Size"),
        ("epochs",          "Epochs"),
        ("threshold",       "Threshold"),
        ("use_aug",         "Use Augmentation (true/false)"),
    ]
    print("Would you like to modify any settings? (y/n): ", end="")
    ans = input().strip().lower()
    if ans != "y":
        return cfg

    print("\nFor each setting, press ENTER to keep the current value, or type a new value.")
    for key, label in editable:
        current = cfg[key]
        user_input = input(f"  {label} [{current}]: ").strip()
        if user_input:
            # Type-cast appropriately
            if key in ("learning_rate", "threshold"):
                cfg[key] = float(user_input)
            elif key in ("batch_size", "epochs"):
                cfg[key] = int(user_input)
            elif key == "use_aug":
                cfg[key] = user_input.lower() in ("true", "1", "yes", "y")
            else:
                cfg[key] = resolve_path(user_input)
    return cfg


def run_headless_mode(config_path: str):
    """Interactive CLI loop that mirrors the GUI functionality."""
    import subprocess

    cfg = load_or_create_config(config_path)
    print_banner()
    print_config(cfg)

    while True:
        print("Select mode:")
        print("  [1] Train")
        print("  [2] Test")
        print("  [3] Predict")
        print("  [4] Show current config")
        print("  [5] Modify config")
        print("  [q] Quit")
        choice = input(">>> ").strip().lower()

        if choice == "q":
            print("Bye!")
            break

        elif choice == "4":
            print_config(cfg)
            continue

        elif choice == "5":
            cfg = ask_modify_config(cfg)
            save_config(cfg, config_path)
            print("Configuration saved.\n")
            print_config(cfg)
            continue

        elif choice == "1":
            # ---- TRAIN ----
            cfg = ask_modify_config(cfg)
            save_config(cfg, config_path)
            args = [
                sys.executable, "-u",
                os.path.join(SCRIPT_DIR, "unet_segmentation_sigmoid_aug.py"),
                "train",
                "--train_images", cfg["train_images"],
                "--train_labels", cfg["train_labels"],
                "--val_images",   cfg["val_images"],
                "--val_labels",   cfg["val_labels"],
                "--test_images",  cfg["test_images"],
                "--test_labels",  cfg["test_labels"],
                "--lr",           str(cfg["learning_rate"]),
                "--batch_size",   str(cfg["batch_size"]),
                "--epochs",       str(cfg["epochs"]),
                "--threshold",    str(cfg["threshold"]),
            ]
            if cfg.get("use_aug"):
                args.append("--use_aug")
            print(f"\n[Running] {' '.join(args)}\n")
            subprocess.run(args, cwd=SCRIPT_DIR)

        elif choice == "2":
            # ---- TEST ----
            cfg = ask_modify_config(cfg)
            save_config(cfg, config_path)
            if not cfg["model_path"]:
                print("[ERROR] model_path is not set. Please set it first (option 5).\n")
                continue
            args = [
                sys.executable, "-u",
                os.path.join(SCRIPT_DIR, "unet_segmentation_sigmoid_aug.py"),
                "test",
                "--test_images", cfg["test_images"],
                "--test_labels", cfg["test_labels"],
                "--model_path",  cfg["model_path"],
                "--threshold",   str(cfg["threshold"]),
            ]
            if cfg.get("use_aug"):
                args.append("--use_aug")
            print(f"\n[Running] {' '.join(args)}\n")
            subprocess.run(args, cwd=SCRIPT_DIR)

        elif choice == "3":
            # ---- PREDICT ----
            cfg = ask_modify_config(cfg)
            save_config(cfg, config_path)
            if not cfg["model_path"]:
                print("[ERROR] model_path is not set. Please set it first (option 5).\n")
                continue
            args = [
                sys.executable, "-u",
                os.path.join(SCRIPT_DIR, "unet_segmentation_sigmoid_aug.py"),
                "predict",
                "--predict_images", cfg["predict_images"],
                "--output_tif",     cfg["output_tif"],
                "--model_path",     cfg["model_path"],
                "--threshold",      str(cfg["threshold"]),
            ]
            if cfg.get("use_aug"):
                args.append("--use_aug")
            print(f"\n[Running] {' '.join(args)}\n")
            subprocess.run(args, cwd=SCRIPT_DIR)

        else:
            print("Invalid choice. Please try again.\n")


# ---------------------------------------------------------------------------
# ================================  GUI MODE  ===============================
# ---------------------------------------------------------------------------

def run_gui_mode(config_path: str):
    """Launch the original PyQt5 GUI."""
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget,
        QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
        QLabel, QTextEdit, QFileDialog, QMessageBox, QCheckBox
    )
    from PyQt5.QtCore import QProcess, Qt
    from PyQt5.QtGui import QPixmap, QDoubleValidator, QIntValidator

    class UNetGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("U-Net Segmentation Toolbox")
            self.config_path = config_path
            self.config = load_or_create_config(self.config_path)

            # QProcess for running the script
            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self._append_output)
            self.process.readyReadStandardError.connect(self._append_output)
            self.process.setProcessChannelMode(QProcess.MergedChannels)

            self._init_ui()

        def _save_config(self):
            save_config(self.config, self.config_path)

        def _init_ui(self):
            main = QWidget()
            layout = QVBoxLayout()
            layout.setContentsMargins(5, 5, 5, 5)
            layout.setSpacing(10)

            # Logo (left-top)
            logo_path = os.path.join(SCRIPT_DIR, "logo.png")
            if os.path.exists(logo_path):
                top_bar = QWidget()
                top_layout = QHBoxLayout()
                top_layout.setContentsMargins(0, 0, 0, 0)
                top_layout.setSpacing(0)

                logo_label = QLabel()
                pixmap = QPixmap(logo_path)
                if pixmap.height() > 64:
                    pixmap = pixmap.scaledToHeight(64, Qt.SmoothTransformation)
                logo_label.setPixmap(pixmap)
                logo_label.setFixedSize(pixmap.size())
                top_layout.addWidget(logo_label)
                top_layout.addStretch()
                top_bar.setLayout(top_layout)
                layout.addWidget(top_bar)

            # Tabs
            tabs = QTabWidget()
            tabs.addTab(self._create_train_tab(), "Train")
            tabs.addTab(self._create_test_tab(), "Test")
            tabs.addTab(self._create_predict_tab(), "Predict")
            layout.addWidget(tabs)

            # Log output
            self.log_output = QTextEdit()
            self.log_output.setReadOnly(True)
            layout.addWidget(QLabel("Output Log:"))
            layout.addWidget(self.log_output)

            main.setLayout(layout)
            self.setCentralWidget(main)

        def _append_output(self):
            output = self.process.readAll().data().decode()
            self.log_output.append(output)

        def _browse_folder(self, key, edt):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            path = QFileDialog.getExistingDirectory(self, "Select Folder", "", options)
            if path:
                self.config[key] = path
                edt.setText(path)
                self._save_config()

        def _browse_file(self, key, edt, caption="Select File", file_filter="*"):
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog

            if key == "output_tif":
                path, _ = QFileDialog.getSaveFileName(self, caption, "", file_filter, "", options)
            else:
                path, _ = QFileDialog.getOpenFileName(self, caption, "", file_filter, "", options)

            if path:
                self.config[key] = path
                edt.setText(path)
                self._save_config()

        def _create_labeled_browse(self, label_text, config_key, is_file=False, file_filter="*"):
            h = QHBoxLayout()
            lbl = QLabel(label_text)
            edt = QLineEdit(str(self.config.get(config_key, "")))
            btn = QPushButton("Browse")
            if is_file:
                btn.clicked.connect(lambda: self._browse_file(config_key, edt, caption=f"Select {label_text}", file_filter=file_filter))
            else:
                btn.clicked.connect(lambda: self._browse_folder(config_key, edt))
            h.addWidget(lbl)
            h.addWidget(edt)
            h.addWidget(btn)
            return h, edt

        def _create_train_tab(self):
            w = QWidget()
            l = QVBoxLayout()
            # File selectors: only train/val
            for label, key in [
                ("Train Images:", "train_images"), ("Train Labels:", "train_labels"),
                ("Val Images:", "val_images"),     ("Val Labels:", "val_labels")
            ]:
                row, _ = self._create_labeled_browse(label, key)
                l.addLayout(row)

            # Hyperparameters
            h_lr = QHBoxLayout()
            h_lr.addWidget(QLabel("Learning Rate (--lr):"))
            self.lr_edit = QLineEdit(str(self.config.get("learning_rate", 0.001)))
            self.lr_edit.setValidator(QDoubleValidator(1e-6, 10.0, 6, self))
            self.lr_edit.setFixedWidth(100)
            self.lr_edit.editingFinished.connect(self._on_lr_changed)
            h_lr.addWidget(self.lr_edit)
            h_lr.addStretch()
            l.addLayout(h_lr)

            h_bs = QHBoxLayout()
            h_bs.addWidget(QLabel("Batch Size (--batch-size):"))
            self.bs_edit = QLineEdit(str(self.config.get("batch_size", 8)))
            self.bs_edit.setValidator(QIntValidator(1, 1024, self))
            self.bs_edit.setFixedWidth(80)
            self.bs_edit.editingFinished.connect(self._on_bs_changed)
            h_bs.addWidget(self.bs_edit)
            h_bs.addStretch()
            l.addLayout(h_bs)

            h_ep = QHBoxLayout()
            h_ep.addWidget(QLabel("Epochs (--epochs):"))
            self.ep_edit = QLineEdit(str(self.config.get("epochs", 10)))
            self.ep_edit.setValidator(QIntValidator(1, 10000, self))
            self.ep_edit.setFixedWidth(80)
            self.ep_edit.editingFinished.connect(self._on_ep_changed)
            h_ep.addWidget(self.ep_edit)
            h_ep.addStretch()
            l.addLayout(h_ep)

            # Augmentation toggle
            self.aug_checkbox = QCheckBox("Use Augmentation (--use_aug)")
            self.aug_checkbox.setChecked(self.config.get("use_aug", False))
            self.aug_checkbox.toggled.connect(self._on_toggle_aug)
            l.addWidget(self.aug_checkbox)

            # Run buttons
            self.start_btn = QPushButton("Start Training")
            self.start_btn.clicked.connect(self._on_start_train)
            self.stop_btn = QPushButton("Stop Training")
            self.stop_btn.clicked.connect(self._on_stop)
            self.stop_btn.hide()
            btn_layout = QHBoxLayout()
            btn_layout.addWidget(self.start_btn)
            btn_layout.addWidget(self.stop_btn)
            l.addLayout(btn_layout)

            w.setLayout(l)
            return w

        def _create_test_tab(self):
            w = QWidget()
            l = QVBoxLayout()
            for label, key, is_file, flt in [
                ("Test Images:", "test_images", False, ""),
                ("Test Labels:", "test_labels", False, ""),
                ("Model Path:",  "model_path",  True,  "*.pt")
            ]:
                row, _ = self._create_labeled_browse(label, key, is_file, flt)
                l.addLayout(row)

            btn = QPushButton("Start Testing")
            btn.clicked.connect(self._on_start_test)
            l.addWidget(btn)
            w.setLayout(l)
            return w

        def _create_predict_tab(self):
            w = QWidget()
            l = QVBoxLayout()
            for label, key, is_file, flt in [
                ("Predict Images:", "predict_images", False, ""),
                ("Output TIFF:",    "output_tif",     True, "TIFF Files (*.tif)"),
                ("Model Path:",     "model_path",     True, "*.pt")
            ]:
                row, _ = self._create_labeled_browse(label, key, is_file, flt)
                l.addLayout(row)
            # Threshold input
            h_thresh = QHBoxLayout()
            h_thresh.addWidget(QLabel("Threshold (--threshold):"))
            self.threshold_edit = QLineEdit(str(self.config.get("threshold", 0.5)))
            self.threshold_edit.setValidator(QDoubleValidator(0.0, 1.0, 3, self))
            self.threshold_edit.setFixedWidth(80)
            self.threshold_edit.editingFinished.connect(self._on_threshold_changed)
            h_thresh.addWidget(self.threshold_edit)
            h_thresh.addStretch()
            l.addLayout(h_thresh)

            btn = QPushButton("Start Prediction")
            btn.clicked.connect(self._on_start_predict)
            l.addWidget(btn)
            w.setLayout(l)
            return w

        def _ask_stop(self):
            res = QMessageBox.question(
                self, "Confirm Stop",
                "A process is currently running. Do you want to stop it?",
                QMessageBox.Yes | QMessageBox.No
            )
            return res == QMessageBox.Yes

        # Hyperparameter handlers
        def _on_lr_changed(self):
            try:
                val = float(self.lr_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Learning Rate", "Enter a valid float.")
                self.lr_edit.setText(str(self.config.get("learning_rate", 0.001)))
                return
            self.config["learning_rate"] = val
            self._save_config()

        def _on_bs_changed(self):
            try:
                val = int(self.bs_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Batch Size", "Enter a valid integer.")
                self.bs_edit.setText(str(self.config.get("batch_size", 4)))
                return
            self.config["batch_size"] = val
            self._save_config()

        def _on_ep_changed(self):
            try:
                val = int(self.ep_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Epochs", "Enter a valid integer.")
                self.ep_edit.setText(str(self.config.get("epochs", 10)))
                return
            self.config["epochs"] = val
            self._save_config()

        def _on_toggle_aug(self, checked):
            self.config["use_aug"] = checked
            self._save_config()

        def _on_threshold_changed(self):
            try:
                val = float(self.threshold_edit.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Threshold", "Please enter a number between 0 and 1.")
                self.threshold_edit.setText(str(self.config.get("threshold", 0.5)))
                return
            self.config["threshold"] = val
            self._save_config()

        def _on_start_train(self):
            if self.process.state() != QProcess.NotRunning:
                QMessageBox.warning(self, "Training", "Training is already in progress.")
                return
            self.start_btn.setEnabled(False)
            self.stop_btn.show()
            cfg = self.config
            args = [
                "--train_images", cfg['train_images'],
                "--train_labels", cfg['train_labels'],
                "--val_images",   cfg['val_images'],
                "--val_labels",   cfg['val_labels'],
                "--test_images",  cfg.get('test_images',''),
                "--test_labels",  cfg.get('test_labels',''),
                "--lr", str(cfg.get("learning_rate", 0.001)),
                "--batch_size", str(cfg.get("batch_size", 4)),
                "--epochs", str(cfg.get("epochs", 10))
            ]
            if cfg.get("use_aug", False):
                args.append("--use_aug")
            self._run_process("train", args)

        def _on_stop(self):
            if self.process.state() != QProcess.NotRunning:
                self.process.kill()
            self.stop_btn.hide()
            self.start_btn.setEnabled(True)

        def _on_start_test(self):
            if self.process.state() != QProcess.NotRunning:
                if not self._ask_stop():
                    return
                self.process.kill()
            cfg = self.config
            args = [
                "--test_images", cfg['test_images'],
                "--test_labels", cfg['test_labels'],
                "--model_path",  cfg['model_path']
            ]
            self._run_process("test", args)

        def _on_start_predict(self):
            if self.process.state() != QProcess.NotRunning:
                if not self._ask_stop():
                    return
                self.process.kill()
            cfg = self.config
            args = [
                "--predict_images", cfg['predict_images'],
                "--output_tif",     cfg['output_tif'],
                "--model_path",     cfg['model_path']
            ]
            # include threshold if set
            th = cfg.get("threshold", None)
            if th is not None:
                args += ["--threshold", str(th)]
            self._run_process("predict", args)

        def _run_process(self, mode, args_list):
            self.log_output.clear()
            cmd = [sys.executable, "-u",
                   os.path.join(SCRIPT_DIR, "unet_segmentation_sigmoid_aug.py"),
                   mode] + args_list
            self.process.start(cmd[0], cmd[1:])
            if not self.process.waitForStarted():
                QMessageBox.critical(self, "Error", "Failed to start process")

    app = QApplication(sys.argv)
    gui = UNetGUI()
    gui.resize(600, 800)
    gui.show()
    sys.exit(app.exec_())


# ---------------------------------------------------------------------------
# ================================  ENTRY POINT  ===========================
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="U-Net Segmentation Toolbox (GUI / Headless)",
        add_help=True,
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run in headless (CLI) mode instead of launching the GUI."
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(SCRIPT_DIR, "dataset_path.yaml"),
        help="Path to the YAML configuration file. (default: unet/dataset_path.yaml)"
    )

    args = parser.parse_args()

    if args.headless:
        run_headless_mode(args.config)
    else:
        run_gui_mode(args.config)
