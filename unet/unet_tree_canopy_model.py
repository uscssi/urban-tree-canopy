import sys
import os
import yaml
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
        self.config_path = "dataset_path.yaml"
        self._load_or_create_config()

        # QProcess for running the script
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._append_output)
        self.process.readyReadStandardError.connect(self._append_output)
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        self._init_ui()

    def _load_or_create_config(self):
        default = {
            "train_images": "", "train_labels": "",
            "val_images": "",   "val_labels": "",
            "predict_images": "", "output_tif": "",
            "model_path": "",
            "use_aug": False,
            "threshold": 0.5,
            # hyperparameters
            "learning_rate": 0.001,
            "batch_size": 8,
            "epochs": 10
        }
        if not os.path.exists(self.config_path):
            with open(self.config_path, 'w') as f:
                yaml.dump(default, f)
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f)

    def _init_ui(self):
        main = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # Logo (left-top)
        logo_path = "logo.png"
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
        cmd = [sys.executable, "-u", "unet_segmentation_sigmoid_aug.py", mode] + args_list
        self.process.start(cmd[0], cmd[1:])
        if not self.process.waitForStarted():
            QMessageBox.critical(self, "Error", "Failed to start process")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = UNetGUI()
    gui.resize(600, 800)
    gui.show()
    sys.exit(app.exec_())
