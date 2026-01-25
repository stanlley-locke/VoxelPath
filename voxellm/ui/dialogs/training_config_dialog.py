from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QPushButton,
    QDialogButtonBox, QTabWidget, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

class TrainingConfigDialog(QDialog):
    """Pre-training configuration dialog with defaults"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Training Configuration")
        self.setMinimumWidth(500)
        
        # Main layout
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("Configure Training Parameters")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(header)
        
        desc = QLabel("Adjust settings or click 'Use Defaults' for recommended values")
        desc.setStyleSheet("color: #888;")
        layout.addWidget(desc)
        
        # Tab widget for organization
        tabs = QTabWidget()
        
        # Basic Settings Tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1000, 100000)
        self.spin_batch_size.setValue(10000)
        self.spin_batch_size.setSuffix(" tokens")
        basic_layout.addRow("Batch Size:", self.spin_batch_size)
        
        self.spin_val_split = QDoubleSpinBox()
        self.spin_val_split.setRange(0, 50)
        self.spin_val_split.setValue(10)
        self.spin_val_split.setSuffix("%")
        basic_layout.addRow("Validation Split:", self.spin_val_split)
        
        self.spin_max_ngram = QSpinBox()
        self.spin_max_ngram.setRange(2, 5)
        self.spin_max_ngram.setValue(5)
        basic_layout.addRow("Max N-gram Level:", self.spin_max_ngram)
        
        tabs.addTab(basic_tab, "Basic")
        
        # Advanced Settings Tab
        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)
        
        self.spin_checkpoint_interval = QSpinBox()
        self.spin_checkpoint_interval.setRange(1, 50)
        self.spin_checkpoint_interval.setValue(5)
        self.spin_checkpoint_interval.setSuffix(" batches")
        advanced_layout.addRow("Checkpoint Interval:", self.spin_checkpoint_interval)
        
        self.spin_early_stopping = QSpinBox()
        self.spin_early_stopping.setRange(1, 20)
        self.spin_early_stopping.setValue(5)
        self.spin_early_stopping.setSuffix(" epochs")
        advanced_layout.addRow("Early Stopping Patience:", self.spin_early_stopping)
        
        self.check_detailed_logs = QCheckBox("Enable detailed per-batch logging")
        self.check_detailed_logs.setChecked(True)
        advanced_layout.addRow("Logging:", self.check_detailed_logs)
        
        self.check_use_smoothing = QCheckBox("Use label smoothing for loss")
        self.check_use_smoothing.setChecked(True)
        advanced_layout.addRow("Loss Calculation:", self.check_use_smoothing)
        
        tabs.addTab(advanced_tab, "Advanced")
        
        layout.addWidget(tabs)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.btn_defaults = QPushButton("Use Defaults")
        self.btn_defaults.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.btn_defaults)
        
        button_layout.addStretch()
        
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        button_layout.addWidget(self.button_box)
        
        layout.addLayout(button_layout)
        
    def reset_to_defaults(self):
        """Reset all values to recommended defaults"""
        self.spin_batch_size.setValue(10000)
        self.spin_val_split.setValue(10)
        self.spin_max_ngram.setValue(5)
        self.spin_checkpoint_interval.setValue(5)
        self.spin_early_stopping.setValue(5)
        self.check_detailed_logs.setChecked(True)
        self.check_use_smoothing.setChecked(True)
        
    def get_config(self):
        """Return configuration dictionary"""
        return {
            'batch_size': self.spin_batch_size.value(),
            'validation_split': self.spin_val_split.value() / 100.0,
            'max_ngram_level': self.spin_max_ngram.value(),
            'checkpoint_interval': self.spin_checkpoint_interval.value(),
            'early_stopping_patience': self.spin_early_stopping.value(),
            'detailed_logging': self.check_detailed_logs.isChecked(),
            'use_label_smoothing': self.check_use_smoothing.isChecked()
        }
