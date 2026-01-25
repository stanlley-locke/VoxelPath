from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QFrame, QLabel, QGroupBox, QTextEdit, QTabWidget, QPushButton,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QCheckBox, QComboBox, QLineEdit, QMenu, QMenuBar,
    QStatusBar, QFileDialog, QMessageBox, QInputDialog, QDialog, QDialogButtonBox,
    QGridLayout, QScrollArea, QApplication, QListWidget, QListWidgetItem,
    QSizePolicy, QProgressDialog
)
from PyQt6.QtCore import Qt, QTimer, QSize, QUrl
from PyQt6.QtGui import QFont, QColor, QIcon, QAction, QDesktopServices
import qdarktheme
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import os
import json
import datetime
import time
import math
import numpy as np
import traceback
import random
import webbrowser
import re

from ..core.engine import VoxelPathProductionEngine
from ..core.structures import TrainingMetrics
from ..core.logger import ResultLogger
from ..workers.training_worker import TrainingWorker
from ..workers.generation_worker import GenerationWorker
from ..workers.server_worker import ServerWorker
from .widgets import TrainingMetricsTableModel, TrainingProgressVisualizer
from .dialogs import TextAnalysisDialog, BatchGenerationDialog, TrainingConfigDialog
from ..workers.loading_worker import ModelLoaderWorker
from ..workers.rag_worker import RagIngestionWorker
from .styles import VoxelTheme
from .help import AboutDialog, DocumentationDialog, InteractiveTutorial

class VoxelPathProductionUI(QMainWindow):
    """Industrial-grade production interface"""
    def __init__(self):
        super().__init__()
        # Set application style
        # (Handled in main.py via VoxelTheme)
            
        # Core engine
        self.engine = VoxelPathProductionEngine()
        
        # Logging System
        self.logger = ResultLogger()
        
        # Worker threads
        self.training_worker = None
        self.generation_worker = None
        
        # UI state
        self.current_trajectory = []
        self.loaded_models = []
        self.last_generated_text = ""
        self.last_generation_metrics = {}
        self.training_metrics_history = []
        self.training_metrics_history = []
        self.validation_results = []
        self.conversation_history = []
        
        # Setup UI
        self.setWindowTitle("VoxelPath Production System v3.1")
        self.resize(1600, 900)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create central widget and main layout
        self._setup_ui()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        # The following line from the instruction is syntactically incorrect and refers to a widget not yet created.
        # It is omitted to ensure a syntactically correct and functional file.
        # if self.cb_auto_rotate.isChecked():
        #      if hasattr(self, 'ax') and self.ax:
        #         self.rotate_visualization()lize system
        self._system_log("System initialized. Ready for production.")
        
        # Update timer for real-time stats
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_status_bar)
        self.stats_timer.start(5000)  # Update every 5 seconds
        
        # Show startup message
        self.status_bar.showMessage("System: Ready | Use File > Load Corpus to begin training")
        
    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("New Engine", self)
        new_action.triggered.connect(self.reset_engine)
        file_menu.addAction(new_action)
        
        load_corpus_action = QAction("Load Training Corpus", self)
        load_corpus_action.triggered.connect(self.load_corpus)
        file_menu.addAction(load_corpus_action)
        
        load_val_action = QAction("Load Validation Corpus", self)
        load_val_action.triggered.connect(self.load_validation_corpus)
        file_menu.addAction(load_val_action)
        
        file_menu.addSeparator()
        
        save_action = QAction("Save Model", self)
        save_action.triggered.connect(self.save_current_model)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Model", self)
        load_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Generated Text", self)
        export_action.triggered.connect(self.export_generated_text)
        
        batch_action = QAction("Batch Generation", self)
        batch_action.triggered.connect(self.open_batch_generation)
        file_menu.addAction(batch_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        optimize_action = QAction("Optimize Memory", self)
        optimize_action.triggered.connect(self.optimize_memory)
        tools_menu.addAction(optimize_action)
        
        analyze_action = QAction("Analyze Generated Text", self)
        analyze_action.triggered.connect(self.analyze_text)
        tools_menu.addAction(analyze_action)
        
        compare_action = QAction("Compare Models", self)
        compare_action.triggered.connect(self.compare_models_dialog)
        tools_menu.addAction(compare_action)
        
        tools_menu.addSeparator()
        
        viz_action = QAction("3D Trajectory Visualization", self)
        viz_action.triggered.connect(self.show_trajectory_visualization)
        tools_menu.addAction(viz_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        
        docs_action = QAction("Documentation", self)
        docs_action.triggered.connect(self.open_documentation)
        help_menu.addAction(docs_action)
        
        tutorial_action = QAction("Interactive Tutorial", self)
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)

    def compare_models_dialog(self):
        """Open dialog to compare multiple selected models"""
        # Get selected models
        selected_items = self.list_models.selectedItems()
        if len(selected_items) < 2:
            QMessageBox.warning(self, "Selection Error", 
                            "Please select at least 2 models to compare (hold Ctrl to select multiple).")
            return
        
        model_paths = []
        model_names = []
        for item in selected_items:
            model_path = item.data(Qt.ItemDataRole.UserRole)
            model_paths.append(model_path)
            model_names.append(os.path.basename(model_path))
        
        try:
            # Compare models using engine's comparison function
            comparison = self.engine.compare_models(model_paths)
        
            # Create comparison dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Model Comparison")
            dialog.setMinimumSize(800, 600)
        
            layout = QVBoxLayout()
        
            # Create table for comparison
            table = QTableWidget(len(comparison), 6)
            table.setHorizontalHeaderLabels([
                "Model Name", "Tokens Processed", "2-grams", "3-grams", 
                "Memory (MB)", "Training Time (s)"
            ])
        
            row = 0
            for model_name, stats in comparison.items():
                table.setItem(row, 0, QTableWidgetItem(model_name))
            
                if "error" in stats:
                    table.setItem(row, 1, QTableWidgetItem(f"Error: {stats['error']}"))
                    row += 1
                    continue
                
                table.setItem(row, 1, QTableWidgetItem(f"{stats['tokens_processed']:,}"))
            
                # N-gram counts
                ngrams = stats.get("ngrams_learned", {})
                table.setItem(row, 2, QTableWidgetItem(f"{ngrams.get(2, 0):,}"))
                table.setItem(row, 3, QTableWidgetItem(f"{ngrams.get(3, 0):,}"))
            
                table.setItem(row, 4, QTableWidgetItem(f"{stats.get('memory_usage', 0):.1f}"))
                table.setItem(row, 5, QTableWidgetItem(f"{stats.get('training_time', 0):.1f}"))
            
                row += 1
            
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
            layout.addWidget(table)
        
            # Add close button
            button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.accept)
            layout.addWidget(button_box)
        
            dialog.setLayout(layout)
            dialog.exec()
        
        except Exception as e:
            QMessageBox.critical(self, "Comparison Error", f"Failed to compare models:\n{str(e)}")
            self._system_log(f"Comparison error: {str(e)}", "ERROR")
        
    def _setup_ui(self):
        """Setup complete production interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Control Center
        left_panel = self._create_control_panel()
        
        # Right Panel: Visualization & Output
        right_panel = self._create_visualization_panel()
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([500, 1100])
        main_layout.addWidget(splitter)
        
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with all configuration options"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("VOXEL PATH")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color: {VoxelTheme.get_color('accent_secondary')}; padding: 10px; background-color: {VoxelTheme.get_color('bg_panel')}; border-radius: 6px;")
        layout.addWidget(header)
        
        # System Console
        console_group = QGroupBox("System Console")
        console_layout = QVBoxLayout()
        self.txt_console = QTextEdit()
        self.txt_console.setReadOnly(True)
        self.txt_console.setFont(QFont("Consolas", 9))
        self.txt_console.setFont(QFont("Consolas", 9))
        self.txt_console.setMaximumHeight(150)
        console_layout.addWidget(self.txt_console)
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        # Tab Widget for organized controls
        self.tab_widget = QTabWidget()
        
        # Tab 1: Training
        training_tab = self._create_training_tab()
        self.tab_widget.addTab(training_tab, "Training")
        
        # Tab 2: Generation
        generation_tab = self._create_generation_tab()
        self.tab_widget.addTab(generation_tab, "Generation")
        
        # Tab 3: Knowledge Base (RAG)
        knowledge_tab = self._create_knowledge_tab()
        self.tab_widget.addTab(knowledge_tab, "Knowledge Base")
        
        # Tab 3: Chat Interface
        chat_tab = self._create_chat_tab()
        self.tab_widget.addTab(chat_tab, "Chat")
        
        # Tab 3: Model Management
        model_tab = self._create_model_tab()
        self.tab_widget.addTab(model_tab, "Models")
        
        # Tab 4: Analytics
        analytics_tab = self._create_analytics_tab()
        self.tab_widget.addTab(analytics_tab, "Analytics")

        # Tab 5: Server API (New)
        server_tab = self._create_server_tab()
        self.tab_widget.addTab(server_tab, "Server API")
        
        # Tab 6: Settings
        settings_tab = self._create_settings_tab()
        self.tab_widget.addTab(settings_tab, "Tools")
        
        layout.addWidget(self.tab_widget)
        
        return panel

# END OF PART 1

    def _create_training_tab(self) -> QWidget:
        """Create training configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Corpus Selection
        corpus_group = QGroupBox("Training Data")
        corpus_layout = QVBoxLayout()
        
        # Training corpus
        train_layout = QHBoxLayout()
        self.btn_load_corpus = QPushButton("Load Training Corpus")
        self.btn_load_corpus.setMinimumHeight(30)
        self.btn_load_corpus.clicked.connect(self.load_corpus)
        train_layout.addWidget(self.btn_load_corpus)
        self.lbl_corpus = QLabel("None")
        self.lbl_corpus.setStyleSheet("color: #888888;")
        train_layout.addWidget(self.lbl_corpus)
        corpus_layout.addLayout(train_layout)
        
        # Validation corpus
        val_layout = QHBoxLayout()
        self.btn_load_val = QPushButton("Load Validation Corpus")
        self.btn_load_val.setMinimumHeight(30)
        self.btn_load_val.clicked.connect(self.load_validation_corpus)
        val_layout.addWidget(self.btn_load_val)
        self.lbl_val_corpus = QLabel("None")
        self.lbl_val_corpus.setStyleSheet(f"color: {VoxelTheme.get_color('text_secondary')};")
        val_layout.addWidget(self.lbl_val_corpus)
        corpus_layout.addLayout(val_layout)
        
        corpus_group.setLayout(corpus_layout)
        layout.addWidget(corpus_group)
        
        
        # Training Parameters
        param_group = QGroupBox("Training Parameters")
        param_layout = QGridLayout()
        
        # Batch size
        param_layout.addWidget(QLabel("Batch Size:"), 0, 0)
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1000, 500000)
        self.spin_batch_size.setValue(50000)
        self.spin_batch_size.setSuffix(" characters")
        param_layout.addWidget(self.spin_batch_size, 0, 1)
        
        # Memory limit
        param_layout.addWidget(QLabel("Memory Limit:"), 1, 0)
        self.spin_memory_limit = QSpinBox()
        self.spin_memory_limit.setRange(100, 32000)
        self.spin_memory_limit.setValue(4000)
        self.spin_memory_limit.setSuffix(" MB")
        param_layout.addWidget(self.spin_memory_limit, 1, 1)
        
        # Early stopping
        param_layout.addWidget(QLabel("Early Stopping Patience:"), 2, 0)
        self.spin_early_stopping = QSpinBox()
        self.spin_early_stopping.setRange(1, 20)
        self.spin_early_stopping.setValue(5)
        param_layout.addWidget(self.spin_early_stopping, 2, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Training Controls
        control_group = QGroupBox("Training Controls")
        control_layout = QVBoxLayout()
        self.btn_start_training = QPushButton("Start Training")
        self.btn_start_training.setMinimumHeight(40)
        self.btn_start_training.setMinimumHeight(40)
        self.btn_start_training.setProperty("class", "primary_button")
        self.btn_start_training.clicked.connect(self.start_training)
        self.btn_stop_training = QPushButton("Stop Training")
        self.btn_stop_training.setMinimumHeight(30)
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.clicked.connect(self.stop_training)
        self.btn_reset_engine = QPushButton("Reset Engine")
        self.btn_reset_engine.setMinimumHeight(30)
        self.btn_reset_engine.setProperty("class", "danger_button")
        self.btn_reset_engine.clicked.connect(self.reset_engine)
        control_layout.addWidget(self.btn_start_training)
        control_layout.addWidget(self.btn_stop_training)
        control_layout.addWidget(self.btn_reset_engine)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Progress Display
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        self.lbl_training_status = QLabel("Idle - Load corpus to begin")
        self.lbl_training_status.setStyleSheet(f"font-weight: bold; color: {VoxelTheme.get_color('text_secondary')};")
        self.progress_training = QProgressBar()
        self.progress_training.setRange(0, 100)
        self.progress_training.setTextVisible(True)
        self.progress_training.setStyleSheet(f"""
        QProgressBar {{
            border: 1px solid {VoxelTheme.get_color('border')};
            background-color: {VoxelTheme.get_color('bg_input')};
            height: 20px;
            color: {VoxelTheme.get_color('text_primary')};
            text-align: center;
        }}
        QProgressBar::chunk {{
            background-color: {VoxelTheme.get_color('accent_primary')};
            border-radius: 2px;
        }}
        """)
        self.lbl_training_stats = QLabel("No training in progress")
        self.lbl_training_stats.setStyleSheet(f"color: {VoxelTheme.get_color('text_secondary')}; font-size: 10pt;")
        progress_layout.addWidget(self.lbl_training_status)
        progress_layout.addWidget(self.progress_training)
        progress_layout.addWidget(self.lbl_training_stats)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Training metrics table
        metrics_group = QGroupBox("Recent Training Metrics")
        metrics_layout = QVBoxLayout()
        self.table_training_metrics = QTableWidget(0, 11)
        self.table_training_metrics.setHorizontalHeaderLabels([
            'Batch', 'Tokens', '2-grams', '3-grams', '4-grams', '5-grams', 
            'Perplexity', 'Coherence', 'Diversity', 'Memory (MB)', 'Rate (tok/s)'
        ])
        self.table_training_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table_training_metrics.verticalHeader().setVisible(False)
        self.table_training_metrics.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_training_metrics.setFont(QFont("Consolas", 9))
        self.table_training_metrics.setAlternatingRowColors(True)
        metrics_layout.addWidget(self.table_training_metrics)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        # Spacer
        layout.addStretch()
        
        return tab
        
    def _create_generation_tab(self) -> QWidget:
        """Create text generation tab - Redesigned for Workflow"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # 1. Input Section (Seed)
        input_group = QGroupBox("1. Input Context")
        input_layout = QVBoxLayout()
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text to start generation (e.g., 'In the beginning')...")
        self.txt_seed.setMaximumHeight(80)
        self.txt_seed.setFont(QFont("Consolas", 10))
        input_layout.addWidget(self.txt_seed)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # 2. Configuration Grid
        config_group = QGroupBox("2. Generation Parameters")
        config_layout = QGridLayout()
        config_layout.setContentsMargins(10, 15, 10, 10)
        config_layout.setVerticalSpacing(15)
        
        # Row 0: Basic Params
        config_layout.addWidget(QLabel("Output Length:"), 0, 0)
        self.spin_output_length = QSpinBox()
        self.spin_output_length.setRange(50, 5000)
        self.spin_output_length.setValue(300)
        config_layout.addWidget(self.spin_output_length, 0, 1)
        
        config_layout.addWidget(QLabel("Temperature:"), 0, 2)
        self.slider_temp = QSlider(Qt.Orientation.Horizontal)
        self.slider_temp.setRange(10, 200)
        self.slider_temp.setValue(85)
        self.slider_temp.valueChanged.connect(self._update_temp_label)
        config_layout.addWidget(self.slider_temp, 0, 3)
        self.lbl_temp_value = QLabel("0.85")
        config_layout.addWidget(self.lbl_temp_value, 0, 4)
        
        # Row 1: Advanced Params
        config_layout.addWidget(QLabel("Top-P Sampling:"), 1, 2)
        self.slider_top_p = QSlider(Qt.Orientation.Horizontal)
        self.slider_top_p.setRange(10, 100)
        self.slider_top_p.setValue(90)
        self.slider_top_p.valueChanged.connect(self._update_top_p_label)
        config_layout.addWidget(self.slider_top_p, 1, 3)
        self.lbl_top_p_value = QLabel("0.90")
        config_layout.addWidget(self.lbl_top_p_value, 1, 4)
        
        config_layout.addWidget(QLabel("Repetition Penalty:"), 1, 0)
        self.slider_rep_penalty = QSlider(Qt.Orientation.Horizontal)
        self.slider_rep_penalty.setRange(5, 30)
        self.slider_rep_penalty.setValue(12)
        self.slider_rep_penalty.valueChanged.connect(self._update_rep_penalty_label)
        config_layout.addWidget(self.slider_rep_penalty, 1, 1)
        self.lbl_rep_penalty_value = QLabel("1.2")
        # config_layout.addWidget(self.lbl_rep_penalty_value, 1, 2) # Overlap issue in grid, handled differently or just tooltip
        
        # Features Toggles
        features_layout = QHBoxLayout()
        self.cb_use_momentum = QCheckBox("Geometric Momentum")
        self.cb_use_momentum.setChecked(True)
        self.cb_use_astar = QCheckBox("A* Pathfinding")
        self.cb_use_tunnels = QCheckBox("Trajectory Tunnels")
        self.cb_use_tunnels.setChecked(True)
        
        features_layout.addWidget(self.cb_use_momentum)
        features_layout.addWidget(self.cb_use_astar)
        features_layout.addWidget(self.cb_use_tunnels)
        features_layout.addStretch()
        
        config_layout.addLayout(features_layout, 2, 0, 1, 5)
        
        # Min Length hidden/bundled
        self.spin_min_length = QSpinBox()
        self.spin_min_length.setRange(10, 1000)
        self.spin_min_length.setValue(50)
        self.spin_min_length.setVisible(False) 
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # 3. Action Area
        action_layout = QHBoxLayout()
        self.btn_generate = QPushButton("GENERATE TEXT")
        self.btn_generate.setMinimumHeight(45)
        self.btn_generate.setProperty("class", "primary_button") # Use theme class
        self.btn_generate.clicked.connect(self.generate_text)
        
        self.btn_clear_output = QPushButton("Clear")
        self.btn_clear_output.setFixedWidth(80)
        self.btn_clear_output.clicked.connect(self.clear_output)
        
        action_layout.addWidget(self.btn_generate, 5)
        action_layout.addWidget(self.btn_clear_output, 1)
        layout.addLayout(action_layout)
        
        # 4. Output Area
        output_group = QGroupBox("Generated Output")
        output_layout = QVBoxLayout()
        self.txt_output = QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QFont("Georgia", 11))
        # Theme handles styling
        output_layout.addWidget(self.txt_output)
        
        # Analysis Toolbar
        tool_layout = QHBoxLayout()
        self.btn_analyze = QPushButton("Detailed Analysis")
        self.btn_analyze.clicked.connect(self.analyze_text)
        tool_layout.addWidget(self.btn_analyze)
        tool_layout.addStretch()
        
        output_layout.addLayout(tool_layout)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group, 1) # Give it stretch
        
        return tab

    def _create_chat_tab(self) -> QWidget:
        """Create conversational AI interface"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Chat History Area
        history_group = QGroupBox("Conversation History")
        history_layout = QVBoxLayout()
        
        self.chat_history_area = QTextEdit()
        self.chat_history_area.setReadOnly(True)
        self.chat_history_area.setFont(QFont("Segoe UI", 10))
        # Custom styling for chat
        self.chat_history_area.setStyleSheet(f"""
            background-color: {VoxelTheme.get_color('bg_input')};
            color: {VoxelTheme.get_color('text_main')};
            border: 1px solid #444444;
            border-radius: 4px;
            padding: 10px;
        """)
        history_layout.addWidget(self.chat_history_area)
        history_group.setLayout(history_layout)
        layout.addWidget(history_group, 4) # Take up most space
        
        # Chat Input Area
        input_group = QGroupBox("Your Message")
        input_layout = QHBoxLayout()
        
        self.txt_chat_input = QTextEdit()
        self.txt_chat_input.setMaximumHeight(60)
        self.txt_chat_input.setPlaceholderText("Type a message to chat with the AI...")
        self.txt_chat_input.setFont(QFont("Segoe UI", 10))
        
        self.cb_raw_mode = QCheckBox("Raw Conversation Mode (for transcript corpora)")
        self.cb_raw_mode.setToolTip("Check this if your model was trained on raw transcripts without 'User:'/'AI:' prefixes.")
        
        self.btn_send_chat = QPushButton("Send")
        self.btn_send_chat.setMinimumHeight(60)
        self.btn_send_chat.setFixedWidth(80)
        self.btn_send_chat.setProperty("class", "primary_button")
        self.btn_send_chat.clicked.connect(self.send_chat_message)
        
        input_layout.addWidget(self.txt_chat_input)
        
        # Model Selector
        self.combo_model_selector = QComboBox()
        self.combo_model_selector.addItems(["Auto (Ensemble)", "Medical", "Coding", "Creative"])
        self.combo_model_selector.setToolTip("Select specific model expert or let Ensemble decide.")
        input_layout.addWidget(self.combo_model_selector)
        
        input_layout.addWidget(self.cb_raw_mode)
        input_layout.addWidget(self.btn_send_chat)
        input_group.setLayout(input_layout)
        layout.addWidget(input_group, 1)
        
        # Initial greeting
        self.chat_history_area.append(f"<span style='color: #4CAF50;'><b>AI:</b></span> <span style='color: #ffffff;'>Hello! I am VoxelLM. How can I help you today?</span><br>")
        
        return tab

    def send_chat_message(self):
        """Handle sending chat message"""
        user_text = self.txt_chat_input.toPlainText().strip()
        if not user_text:
            return
            
        # Display user message
        timestamp = datetime.datetime.now().strftime("%H:%M")
        self.chat_history_area.append(f"<span style='color: #2196F3;'><b>You ({timestamp}):</b></span> <span style='color: #ffffff;'>{user_text}</span><br>")
        self.txt_chat_input.clear()
        
        # Add to history
        self.conversation_history.append({"role": "User", "content": user_text})
        
        # Prepare Prompt from History
        if self.cb_raw_mode.isChecked():
            # Raw Mode: Simple newlines, no role prefixes
            # Limit history to prevent context overflow (simpler logic)
            prompt = ""
            # Don't include the just-added current message in the loop, wait..
            # History has it. So just iterate history.
            for msg in self.conversation_history[-8:]:
               prompt += f"{msg['content']}\n" 
            # Current user text is already in history, so loop covers it.
            
            stop_candidates = ["\n"] # Stop after one line
        else:
            # Standard Mode: User/AI Prefixes
            # Limit to last 6 turns to fit context and avoid overflowing
            recent_history = self.conversation_history[-6:]
            prompt = "The following is a conversation between a User and an AI Assistant.\n"
            for msg in recent_history:
                # Normal mode needs the prefix construction
                if msg['role'] == "User": # Don't double add if logic is reused? No, history is source of truth.
                     pass 
                prompt += f"{msg['role']}: {msg['content']}\n"
            prompt += "AI:"
            stop_candidates = ["User:", "\nUser", "User "]

        # Get parameters from Generation Tab
        temperature = self.slider_temp.value() / 100.0
        rep_penalty = self.slider_rep_penalty.value() / 10.0
        top_p = self.slider_top_p.value() / 100.0
        
        # Generate response
        try:
            self.btn_send_chat.setEnabled(False)
            self.status_bar.showMessage("AI is typing...")
            QApplication.processEvents()
            
            generated_text, _, _ = self.engine.generate_text(
                seed_text=prompt,
                max_length=200,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                stop_sequences=stop_candidates,
                use_momentum=True
            )
            
            # Simple cleanup to prevent AI from roleplaying user if stop sequence fails
            response = generated_text
            
            # Robust prompt stripping
            # Remove the prompt if it was returned (engine usually returns full text? No, generate_text appends. Wait.)
            # Engine logic: generated_tokens = seed_tokens.copy() ... return tokenizer.detokenize(generated_tokens)
            # So engine returns PROMPT + COMPLETION.
            # I need to strip the prompt.
            
            # Robust prompt stripping
            # 1. Strict match
            if response.startswith(prompt):
                 response = response[len(prompt):]
            # 2. Case-insensitive match (common due to tokenization)
            elif response.lower().startswith(prompt.lower()):
                 response = response[len(prompt):]
            
            # 3. Fallback: Anchor Search (Robust against missing newlines/mashed text)
            # Find the user's input in the response, then find the next "AI:"
            if len(response) > len(prompt) + 10:
                # Find the LAST occurrence of the user text (in case of repetition)
                last_user_idx = response.rfind(user_text)
                
                if last_user_idx != -1:
                    # Look at text AFTER the user input
                    after_user = response[last_user_idx + len(user_text):]
                    
                    # Find the first "AI:" or "ai:" in this segment
                    # We look for "ai" followed by colon, maybe with some junk in between
                    ai_match = re.search(r"ai\s*:", after_user, re.IGNORECASE)
                    
                    if ai_match:
                        # Everything after this "AI:" is the real response
                        response = after_user[ai_match.end():]
                
                # Double fallback: If we still see "The following is a conversation..."
                if "The following is a conversation" in response:
                     # This usually means stripping failed entirely.
                     # Last ditch: simple known prefix strip
                     known_prefix = "The following is a conversation between a user and an ai assistant."
                     if response.lower().startswith(known_prefix.lower()):
                         # It might be mashed like "...assistant.User:..."
                         # Just hunt for "ai:" generally?
                         pass
            
            # Clean up response
            if "User:" in response:
                response = response.split("User:")[0]
            if "\nUser" in response:
                 response = response.split("\nUser")[0]
            
            response = response.strip()
            
            if not response:
                response = "..."
                
            # Check for repetition of the system prompt
            if "The following is a conversation" in response:
                 response = response.replace("The following is a conversation between a user and an ai assistant.", "")\
                                  .replace("The following is a conversation between a User and an AI Assistant.", "")\
                                  .strip()
            
            # Display AI response with explicit white color to avoid theme issues
            self.chat_history_area.append(f"<span style='color: #4CAF50;'><b>AI:</b></span> <span style='color: #ffffff;'>{response}</span><br>")
            self.chat_history_area.verticalScrollBar().setValue(self.chat_history_area.verticalScrollBar().maximum())
            
            # Add to history
            self.conversation_history.append({"role": "AI", "content": response})
            
        except Exception as e:
            QMessageBox.critical(self, "Chat Error", str(e))
            traceback.print_exc()
        finally:
            self.btn_send_chat.setEnabled(True)
            self.status_bar.showMessage("Ready")

    def _create_model_tab(self) -> QWidget:
        """Create model management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Model Browser
        browser_group = QGroupBox("Saved Models")
        browser_layout = QVBoxLayout()
        self.list_models = QListWidget()
        self.list_models.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list_models.setAlternatingRowColors(True)
        self.list_models.setFont(QFont("Consolas", 9))
        self.list_models.itemDoubleClicked.connect(self.load_selected_model)
        browser_layout.addWidget(self.list_models)
        browser_group.setLayout(browser_layout)
        layout.addWidget(browser_group)
        
        # Model Controls
        model_control_group = QGroupBox("Model Operations")
        model_control_layout = QGridLayout()
        
        self.btn_refresh_models = QPushButton("Refresh List")
        self.btn_refresh_models.clicked.connect(self.refresh_model_list)
        model_control_layout.addWidget(self.btn_refresh_models, 0, 0)
        
        self.btn_load_model = QPushButton("Load Selected Model")
        self.btn_load_model.clicked.connect(self.load_selected_model)
        model_control_layout.addWidget(self.btn_load_model, 0, 1)
        
        self.btn_delete_model = QPushButton("Delete Selected Model")
        self.btn_delete_model.setProperty("class", "danger_button")
        self.btn_delete_model.clicked.connect(self.delete_selected_model)
        model_control_layout.addWidget(self.btn_delete_model, 1, 0)
        
        self.btn_export_model = QPushButton("Export Model")
        self.btn_export_model.clicked.connect(self.export_model)
        model_control_layout.addWidget(self.btn_export_model, 1, 1)
        
        self.btn_compare_models = QPushButton("Compare Selected Models")
        self.btn_compare_models.clicked.connect(self.compare_selected_models)
        model_control_layout.addWidget(self.btn_compare_models, 2, 0, 1, 2)
        
        model_control_group.setLayout(model_control_layout)
        layout.addWidget(model_control_group)
        
        # Model Information
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        self.txt_model_info = QTextEdit()
        self.txt_model_info.setReadOnly(True)
        self.txt_model_info.setMaximumHeight(150)
        self.txt_model_info.setFont(QFont("Consolas", 9))
        info_layout.addWidget(self.txt_model_info)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Refresh model list
        self.refresh_model_list()
        
        return tab

    def _create_analytics_tab(self) -> QWidget:
        """Create analytics and visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        
        # Training Metrics Visualization
        viz_group = QGroupBox("Training Metrics Visualization")
        viz_layout = QVBoxLayout()
        
        self.training_viz = TrainingProgressVisualizer()
        viz_layout.addWidget(self.training_viz)
        
        # Controls
        control_layout = QHBoxLayout()
        self.btn_refresh_viz = QPushButton("Refresh Visualization")
        self.btn_refresh_viz.clicked.connect(self.refresh_training_visualization)
        self.btn_clear_viz = QPushButton("Clear Visualization")
        self.btn_clear_viz.clicked.connect(self.clear_training_visualization)
        self.btn_export_viz = QPushButton("Export Metrics")
        self.btn_export_viz.clicked.connect(self.export_training_metrics)
        
        control_layout.addWidget(self.btn_refresh_viz)
        control_layout.addWidget(self.btn_clear_viz)
        control_layout.addWidget(self.btn_export_viz)
        control_layout.addStretch()
        
        viz_layout.addLayout(control_layout)
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Quality Metrics
        quality_group = QGroupBox("Text Quality Benchmarks")
        quality_layout = QVBoxLayout()
        
        self.table_quality_metrics = QTableWidget(0, 3)
        self.table_quality_metrics.setHorizontalHeaderLabels(["Metric", "Value", "Benchmark"])
        self.table_quality_metrics.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_quality_metrics.verticalHeader().setVisible(False)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)
        
        return tab

    def _create_settings_tab(self) -> QWidget:
        """Create settings and tools tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        
        # Appearance
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout()
        
        self.btn_toggle_theme = QPushButton("Toggle Theme (Light/Dark)")
        self.btn_toggle_theme.setMinimumHeight(40)
        self.btn_toggle_theme.clicked.connect(self.toggle_theme)
        appearance_layout.addWidget(self.btn_toggle_theme)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Advanced Tools
        tools_group = QGroupBox("Optimization Tools")
        tools_layout = QVBoxLayout()
        
        btn_optimize = QPushButton("Optimize Memory")
        btn_optimize.clicked.connect(self.optimize_memory)
        tools_layout.addWidget(btn_optimize)
        
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)
        
        layout.addStretch()
        return tab

    def toggle_theme(self):
        """Switch between light and dark themes"""
        app = QApplication.instance()
        new_mode = VoxelTheme.toggle(app)
        
        # Update visualization colors
        self.refresh_visualization()
        
        status = f"Theme switched to {new_mode.upper()}"
        self.status_bar.showMessage(status, 3000)
        self._system_log(status)

    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization panel with 3D plot and output"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        main_vlayout = QVBoxLayout(panel)
        main_vlayout.setSpacing(10)
        
        # 3D Voxel Cloud Visualization Group
        viz_group = QGroupBox("3D Voxel Cloud Visualization")
        vlayout = QVBoxLayout()
        
        # 3D Visualization Header
        viz_header = QLabel("Neural Voxel Cloud & Inference Trajectory")
        viz_header_font = QFont("Consolas", 12, QFont.Weight.Bold)
        viz_header.setFont(viz_header_font)
        viz_header.setStyleSheet("color: #00ffff; padding: 5px; background-color: #1a1a2a;")
        viz_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlayout.addWidget(viz_header)
        
        # Visualization Controls
        viz_controls = QHBoxLayout()
        
        # Density control
        viz_controls.addWidget(QLabel("Cloud Density:"))
        self.slider_density = QSlider(Qt.Orientation.Horizontal)
        self.slider_density.setRange(1, 100)
        self.slider_density.setValue(30)
        self.slider_density.valueChanged.connect(self._update_density_label)
        viz_controls.addWidget(self.slider_density)
        self.lbl_density_value = QLabel("30%")
        self.lbl_density_value.setMinimumWidth(40)
        viz_controls.addWidget(self.lbl_density_value)
        
        # Color map selection
        viz_controls.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'jet', 'hsv'])
        self.combo_colormap.setCurrentText('viridis')
        self.combo_colormap.currentTextChanged.connect(self.refresh_visualization)
        viz_controls.addWidget(self.combo_colormap)
        
        # Point size
        viz_controls.addWidget(QLabel("Point Size:"))
        self.spin_point_size = QSpinBox()
        self.spin_point_size.setRange(1, 10)
        self.spin_point_size.setValue(1)
        self.spin_point_size.valueChanged.connect(self.refresh_visualization)
        viz_controls.addWidget(self.spin_point_size)
        
        # Refresh and clear buttons
        self.btn_refresh_viz = QPushButton("Refresh")
        self.btn_refresh_viz.clicked.connect(self.refresh_visualization)
        viz_controls.addWidget(self.btn_refresh_viz)
        
        self.btn_clear_viz = QPushButton("Clear View")
        self.btn_clear_viz.clicked.connect(self.clear_visualization)
        viz_controls.addWidget(self.btn_clear_viz)
        
        viz_controls.addStretch()
        vlayout.addLayout(viz_controls)
        
        # 3D Plot Canvas - White Background for GitHub Theme
        self.fig = Figure(facecolor='white', figsize=(10, 6))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        # Create axis
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('white')
        
        # Set title
        self.ax.set_title('3D Voxel Token Space', color='black', fontsize=12, pad=10)
        self.ax.grid(False)
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        
        self.ax.xaxis.line.set_color("white")
        self.ax.yaxis.line.set_color("white")
        self.ax.zaxis.line.set_color("white")
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        
        # Set labels
        self.ax.set_xlabel('X (Char 1)', color='white', fontsize=10, labelpad=8)
        self.ax.set_ylabel('Y (Char 2)', color='white', fontsize=10, labelpad=8)
        self.ax.set_zlabel('Z (Char 3)', color='white', fontsize=10, labelpad=8)
        
        # Set title
        self.ax.set_title('3D Voxel Token Space', color='white', fontsize=12, pad=10)
        
        vlayout.addWidget(self.canvas)
        
        # 3D Controls
        controls_layout = QHBoxLayout()
        
        self.btn_reset_view = QPushButton("Reset View")
        self.btn_reset_view.setFixedWidth(100)
        self.btn_reset_view.clicked.connect(self.reset_visualization_view)
        
        self.cb_auto_rotate = QCheckBox("Auto-Rotate")
        self.cb_auto_rotate.toggled.connect(self.toggle_auto_rotate)
        
        self.lbl_viz_status = QLabel("Ready")
        self.lbl_viz_status.setStyleSheet(f"color: {VoxelTheme.get_color('text_dim')}; font-size: 9pt;")
        
        controls_layout.addWidget(self.btn_reset_view)
        controls_layout.addWidget(self.cb_auto_rotate)
        controls_layout.addStretch()
        controls_layout.addWidget(self.lbl_viz_status)
        
        vlayout.addLayout(controls_layout)
        
        # Add layout to group
        viz_group.setLayout(vlayout)
        main_vlayout.addWidget(viz_group, 3)
        
        # Output Log (Bottom)
        log_group = QGroupBox("System Activity Log")
        log_layout = QVBoxLayout()
        self.txt_full_output = QTextEdit()
        self.txt_full_output.setReadOnly(True)
        self.txt_full_output.setFont(QFont("Consolas", 10))
        # Styles handled by VoxelTheme
        
        log_layout.addWidget(self.txt_full_output)
        log_group.setLayout(log_layout)
        main_vlayout.addWidget(log_group, 1) # Less vertical space
        
        # Initial visualization
        self.refresh_visualization()
        
        return panel

    def reset_visualization_view(self):
        """Reset camera angle"""
        if hasattr(self, 'ax') and self.ax:
            self.ax.view_init(elev=20, azim=45)
            self.canvas.draw()

    def toggle_auto_rotate(self, enabled):
        """Enable/disable auto-rotation"""
        if enabled:
            if not hasattr(self, 'rotate_timer'):
                self.rotate_timer = QTimer()
                self.rotate_timer.timeout.connect(self.rotate_visualization)
            self.rotate_timer.start(50) # 20 FPS
            self._system_log("Auto-rotation enabled for 3D visualization.", "DEBUG")
        else:
            if hasattr(self, 'rotate_timer'):
                self.rotate_timer.stop()
            self._system_log("Auto-rotation disabled for 3D visualization.", "DEBUG")

    def rotate_visualization(self):
        """Rotate the 3D view slightly"""
        if hasattr(self, 'ax') and self.ax:
            self.ax.azim += 0.5
            self.ax.dist = 10 # Keep distance constant
            self.canvas.draw()

# END OF PART 2

    # UI UPDATE METHODS
    
    def _update_temp_label(self):
        """Update temperature label"""
        value = self.slider_temp.value() / 100.0
        self.lbl_temp_value.setText(f"{value:.2f}")
        
    def _update_top_p_label(self):
        """Update top-p label"""
        value = self.slider_top_p.value() / 100.0
        self.lbl_top_p_value.setText(f"{value:.2f}")
        
    def _update_rep_penalty_label(self):
        """Update repetition penalty label"""
        value = self.slider_rep_penalty.value() / 10.0
        self.lbl_rep_penalty_value.setText(f"{value:.1f}")
        
    def _update_density_label(self):
        """Update density label"""
        value = self.slider_density.value()
        self.lbl_density_value.setText(f"{value}%")
        
    def _system_log(self, message: str, level: str = "INFO"):
        """Log message to system console with timestamp and color coding"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        # Color coding based on level
        if level == "ERROR":
            color = "#ff4444"
            prefix = "ERROR"
        elif level == "WARNING":
            color = "#ffaa00"
            prefix = "WARNING"
        elif level == "SUCCESS":
            color = "#44ff44"
            prefix = "SUCCESS"
        elif level == "DEBUG":
            color = "#aaaaff"
            prefix = "DEBUG"
        else:
            color = "#00aaff"
            prefix = "INFO"
            
        # Format message
        formatted = f'<span style="color: #888888;">[{timestamp}]</span> ' \
                   f'<span style="color: {color}; font-weight: bold;">{prefix}</span>: ' \
                   f'<span style="color: #dddddd;">{message}</span>'
                   
        # Append to console
        self.txt_console.append(formatted)
        
        # Auto-scroll to bottom
        scrollbar = self.txt_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status bar
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"{prefix}: {message}", 5000)
            
    def _update_status_bar(self):
        """Update status bar with current statistics"""
        if hasattr(self, 'engine'):
            uptime = int(time.time() - self.engine.start_time)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            
            status = f"Tokens: {self.engine.tokens_processed:,} | " \
                    f"N-Grams: {sum(self.engine.ngrams_learned.values()):,} | " \
                    f"Memory: {self.engine.memory_usage_mb:.1f} MB | " \
                    f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}"
                    
            self.status_bar.showMessage(status)
            
    def update_training_statistics(self, metrics: dict):
        """Update training statistics display"""
        if not metrics:
            return
            
        # Update table
        row = self.table_training_metrics.rowCount()
        self.table_training_metrics.insertRow(row)
        
        # Add data to row
        items = [
            QTableWidgetItem(str(metrics.get('batch_idx', row) + 1)),
            QTableWidgetItem(f"{metrics.get('tokens_processed', 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(2, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(3, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(4, 0):,}"),
            QTableWidgetItem(f"{metrics.get('ngrams_learned', {}).get(5, 0):,}"),
            QTableWidgetItem(f"{metrics.get('perplexity', float('inf')):.2f}"),
            QTableWidgetItem(f"{metrics.get('coherence_score', 0.0):.2f}"),
            QTableWidgetItem(f"{metrics.get('diversity_score', 0.0):.2f}"),
            QTableWidgetItem(f"{metrics.get('memory_usage_mb', 0.0):.1f}"),
            QTableWidgetItem(f"{metrics.get('training_rate', 0.0):.0f}")
        ]
        
        for col, item in enumerate(items):
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table_training_metrics.setItem(row, col, item)
            
        # Keep only the last 20 rows for performance
        if self.table_training_metrics.rowCount() > 20:
            self.table_training_metrics.removeRow(0)
            
        # Update metrics history for visualization
        self.training_metrics_history.append(metrics)
        self.training_viz.update_metrics(self.training_metrics_history[-50:])  # Keep last 50
        
        # Update quality benchmarks
        self.update_quality_benchmarks(metrics)

    def update_quality_benchmarks(self, metrics: dict):
        """Update quality benchmark table"""
        self.table_quality_metrics.setRowCount(0)
        
        benchmark_data = [
            ("Perplexity", metrics.get('perplexity', float('inf')), "Lower is better (<150 good)"),
            ("Coherence Score", metrics.get('coherence_score', 0.0), "Higher is better (>0.6 good)"),
            ("Diversity Score", metrics.get('diversity_score', 0.0), "Higher is better (>0.4 good)"),
            ("Memory Usage", self.engine.memory_usage_mb, "Lower is better (<2000 MB good)"),
            ("Training Rate", metrics.get('training_rate', 0.0), "Higher is better (>1000 tok/s good)"),
            ("2-grams Learned", self.engine.ngrams_learned.get(2, 0), "More is better"),
            ("5-grams Learned", self.engine.ngrams_learned.get(5, 0), "More is better"),
        ]
        
        for i, (metric, value, benchmark) in enumerate(benchmark_data):
            self.table_quality_metrics.insertRow(i)
            
            # Metric name
            item_metric = QTableWidgetItem(metric)
            self.table_quality_metrics.setItem(i, 0, item_metric)
            
            # Value with conditional formatting
            item_value = QTableWidgetItem(f"{value:.2f}" if isinstance(value, float) else f"{value:,}")
            if metric == "Perplexity":
                if value < 150:
                    item_value.setBackground(QColor(50, 150, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                elif value < 300:
                    item_value.setBackground(QColor(150, 150, 50))
                    item_value.setForeground(QColor(0, 0, 0))
                else:
                    item_value.setBackground(QColor(150, 50, 50))
                    item_value.setForeground(QColor(255, 255, 255))
            elif metric in ["Coherence Score", "Diversity Score"]:
                if value > 0.6:
                    item_value.setBackground(QColor(50, 150, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                elif value > 0.4:
                    item_value.setBackground(QColor(150, 150, 50))
                    item_value.setForeground(QColor(0, 0, 0))
                else:
                    item_value.setBackground(QColor(150, 50, 50))
                    item_value.setForeground(QColor(255, 255, 255))
                    
            self.table_quality_metrics.setItem(i, 1, item_value)
            
            # Benchmark description
            item_bench = QTableWidgetItem(benchmark)
            self.table_quality_metrics.setItem(i, 2, item_bench)
            
    def refresh_training_visualization(self):
        """Refresh the training metrics visualization"""
        self.training_viz.update_metrics(self.training_metrics_history[-100:])  # Show last 100 points
        
    def clear_training_visualization(self):
        """Clear the training visualization"""
        self.training_viz.clear()
        self._system_log("Training visualization cleared")
        
    def export_training_metrics(self):
        """Export training metrics to CSV file"""
        if not self.training_metrics_history:
            QMessageBox.warning(self, "No Data", "No training metrics to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Training Metrics",
            f"training_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    headers = [
                        'timestamp', 'batch_idx', 'tokens_processed', '2_grams', '3_grams', 
                        '4_grams', '5_grams', 'perplexity', 'coherence_score', 
                        'diversity_score', 'memory_usage_mb', 'training_rate', 'batch_time'
                    ]
                    writer.writerow(headers)
                    
                    # Data
                    for metrics in self.training_metrics_history:
                        row = [
                            metrics.get('timestamp', datetime.datetime.now().isoformat()),
                            metrics.get('batch_idx', 0),
                            metrics.get('tokens_processed', 0),
                            metrics.get('ngrams_learned', {}).get(2, 0),
                            metrics.get('ngrams_learned', {}).get(3, 0),
                            metrics.get('ngrams_learned', {}).get(4, 0),
                            metrics.get('ngrams_learned', {}).get(5, 0),
                            metrics.get('perplexity', float('inf')),
                            metrics.get('coherence_score', 0.0),
                            metrics.get('diversity_score', 0.0),
                            metrics.get('memory_usage_mb', 0.0),
                            metrics.get('training_rate', 0.0),
                            metrics.get('batch_time', 0.0)
                        ]
                        writer.writerow(row)
                        
                self._system_log(f"Training metrics exported to: {file_path}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Training metrics saved to:\n{file_path}")
            except Exception as e:
                self._system_log(f"Export error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Export Error", f"Failed to export metrics:\n{str(e)}")

# END OF PART 3
    
    # CORE FUNCTIONALITY
    
    def load_corpus(self):
        """Load training corpus with size validation and preview"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Corpus",
            "",
            "Text Files (*.txt *.csv *.json);;All Files (*.*)"
        )
        if file_path:
            try:
                # Get file size first
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Show preview and get confirmation for large files
                if file_size > 100:
                    reply = QMessageBox.question(
                        self,
                        "Large File Warning",
                        "This file is larger than 100MB. Loading might be slow.\n\n"
                        "Only the first 100MB will be used for initial training. Continue?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                        
                # Read file with size limitation
                max_chars = 100_000_000  # 100MB limit
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    if file_size > 100:
                        # Read only first 100MB
                        corpus_text = f.read(max_chars)
                        self._system_log(f"Loaded first {max_chars:,} characters from large file", "WARNING")
                    else:
                        corpus_text = f.read()
                        
                # Update UI
                self.current_corpus = file_path
                self.lbl_corpus.setText(f"{os.path.basename(file_path)} ({file_size:.1f} MB)")
                
                # Show preview
                preview = corpus_text[:500] + "..." if len(corpus_text) > 500 else corpus_text
                QMessageBox.information(self, "Corpus Loaded", 
                                      f"File: {os.path.basename(file_path)}\n"
                                      f"Size: {file_size:.1f} MB\n"
                                      f"Characters: {len(corpus_text):,}\n\n"
                                      f"Preview:\n{preview}")
                
                self._system_log(f"Corpus loaded: {os.path.basename(file_path)} ({len(corpus_text):,} chars)")
            except Exception as e:
                self._system_log(f"Error loading corpus: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Load Error", f"Failed to load corpus:\n{str(e)}")
                
    def load_validation_corpus(self):
        """Load validation corpus"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Validation Corpus",
            "",
            "Text Files (*.txt *.csv *.json);;All Files (*.*)"
        )
        if file_path:
            try:
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                # Read file with size limitation (smaller for validation)
                max_chars = 1_000_000  # 1MB limit for validation
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    if file_size > 1:
                        validation_text = f.read(max_chars)
                        self._system_log(f"Loaded first {max_chars:,} characters from validation file", "WARNING")
                    else:
                        validation_text = f.read()
                        
                self.validation_corpus = file_path
                self.lbl_val_corpus.setText(f"{os.path.basename(file_path)} ({file_size:.1f} MB)")
                
                self._system_log(f"Validation corpus loaded: {os.path.basename(file_path)}")
            except Exception as e:
                self._system_log(f"Error loading validation corpus: {str(e)}", "ERROR")
                
    def start_training(self):
        """Start training process with configuration dialog"""
        if not hasattr(self, 'current_corpus') or not self.current_corpus:
            QMessageBox.warning(self, "No Corpus", "Please load a training corpus first.")
            return
        
        # Show configuration dialog
        config_dialog = TrainingConfigDialog(self)
        
        if config_dialog.exec() != QDialog.DialogCode.Accepted:
            return  # User cancelled
        
        # Get configuration
        config = config_dialog.get_config()
        self._system_log(f"Training config: {config}", "INFO")
        
        # Reset training history
        self.training_metrics_history = []
        self.validation_results = []
        self.table_training_metrics.setRowCount(0)
        
        # Disable controls during training
        self.btn_start_training.setEnabled(False)
        self.btn_stop_training.setEnabled(True)
        self.btn_load_corpus.setEnabled(False)
        self.btn_load_val.setEnabled(False)
        
        # Reset progress
        self.progress_training.setValue(0)
        self.lbl_training_status.setText("Training in progress...")
        self.lbl_training_stats.setText("Initializing training...")
        
        # Get validation path if available
        validation_path = getattr(self, 'validation_corpus', None)
        
        # Create and start worker with config
        self.training_worker = TrainingWorker(
            self.engine, 
            self.current_corpus, 
            validation_path,
            **config  # Pass configuration
        )
        self.training_worker.progress_updated.connect(self.on_training_progress)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.error_occurred.connect(self.on_training_error)
        self.training_worker.validation_metrics.connect(self.on_validation_metrics)
        self.training_worker.start()
        
        self._system_log("Training started with custom configuration", "INFO")
        
    def on_training_progress(self, progress: int, message: str, metrics: dict):
        """Handle training progress updates"""
        self.progress_training.setValue(progress)
        self.lbl_training_status.setText(message)
        
        if metrics:
            self.lbl_training_stats.setText(
                f"Tokens: {metrics.get('tokens_processed', 0):,} | "
                f"N-Grams: {sum(metrics.get('ngrams_learned', {}).values()):,} | "
                f"Perplexity: {metrics.get('perplexity', float('inf')):.2f}"
            )
            self.update_training_statistics(metrics)
            
    def on_validation_metrics(self, metrics: dict):
        """Handle validation metrics updates"""
        if metrics:
            self.validation_results.append(metrics)
            perplexity = metrics.get('perplexity', float('inf'))
            coherence = metrics.get('coherence_score', 0.0)
            diversity = metrics.get('diversity_score', 0.0)
            
            self._system_log(f"Validation: Perplexity={perplexity:.2f}, "
                           f"Coherence={coherence:.2f}, Diversity={diversity:.2f}", 
                           "SUCCESS")
                           
    def on_training_complete(self, message: str, metrics: dict):
        """Handle training completion"""
        # Re-enable controls
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.btn_load_val.setEnabled(True)
        
        self.lbl_training_status.setText("Training complete")
        self.progress_training.setValue(100)
        
        if metrics:
            final_msg = (
                f"Training completed successfully!\n\n"
                f"Total Tokens Processed: {metrics['tokens_processed']:,}\n"
                f"Total N-Grams Learned: {metrics['total_ngrams']:,}\n"
                f"Final Average Perplexity: {metrics['avg_perplexity']:.2f}\n"
                f"Training Time: {metrics['training_time']:.1f} seconds\n"
                f"Memory Usage: {metrics['memory_usage_mb']:.1f} MB"
            )
            self._system_log(final_msg, "SUCCESS")
            QMessageBox.information(self, "Training Complete", final_msg)
            
        # Update UI
        self.refresh_visualization()
        self.refresh_model_list()
        
        # Show Concept Clouds
        self.show_concepts_dialog()
        
    def on_training_error(self, error_message: str):
        """Handle training errors"""
        self.btn_start_training.setEnabled(True)
        self.btn_stop_training.setEnabled(False)
        self.btn_load_corpus.setEnabled(True)
        self.btn_load_val.setEnabled(True)
        
        self.lbl_training_status.setText("Training failed")
        self.progress_training.setValue(0)
        
        self._system_log(error_message, "ERROR")
        QMessageBox.critical(self, "Training Error", error_message)
        
    def stop_training(self):
        """Stop training process gracefully"""
        if self.training_worker and self.training_worker.isRunning():
            self._system_log("Stopping training process...", "WARNING")
            self.lbl_training_status.setText("Stopping training...")
            
            self.training_worker.stop()
            # Wait with timeout
            if not self.training_worker.wait(5000):  # 5 seconds timeout
                self._system_log("Training stop timed out, forcing termination", "ERROR")
                
            self.btn_start_training.setEnabled(True)
            self.btn_stop_training.setEnabled(False)
            self.lbl_training_status.setText("Training stopped by user")
            self._system_log("Training stopped by user", "WARNING")
            
    def reset_engine(self):
        """Reset the engine to initial state with confirmation"""
        reply = QMessageBox.question(
            self,
            "Reset Engine",
            "This will delete all learned patterns and reset the engine to initial state.\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.engine.reset_engine()
                self.training_metrics_history = []
                self.validation_results = []
                self.table_training_metrics.setRowCount(0)
                self.training_viz.clear()
                
                # Clear outputs
                self.txt_output.clear()
                self.txt_full_output.clear()
                self.current_trajectory = []
                
                # Update UI
                self.refresh_visualization()
                self.refresh_model_list()
                
                self._system_log("Engine reset complete", "SUCCESS")
                QMessageBox.information(self, "Reset Complete", 
                                      "Engine has been reset to initial state.")
            except Exception as e:
                self._system_log(f"Reset error: {str(e)}", "ERROR")
                QMessageBox.critical(self, "Reset Error", f"Failed to reset engine:\n{str(e)}")
                
    def generate_text(self):
        """Generate text based on current parameters"""
        seed_text = self.txt_seed.toPlainText().strip()
        if not seed_text:
            # Use a default seed based on time
            hour = datetime.datetime.now().hour
            if 5 <= hour < 12:
                default_seed = "the morning sun rose over the horizon"
            elif 12 <= hour < 17:
                default_seed = "the afternoon sun cast long shadows"
            elif 17 <= hour < 21:
                default_seed = "the evening stars began to appear"
            else:
                default_seed = "the night was dark and full of secrets"
                
            reply = QMessageBox.question(
                self,
                "No Seed Text",
                f"No seed text entered. Would you like to use this default seed:\n\n'{default_seed}'\n\nOr cancel to enter your own?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                seed_text = default_seed
                self.txt_seed.setPlainText(seed_text)
            elif reply == QMessageBox.StandardButton.Cancel:
                return
                
        # Get parameters
        temperature = self.slider_temp.value() / 100.0
        top_p = self.slider_top_p.value() / 100.0
        repetition_penalty = self.slider_rep_penalty.value() / 10.0
        max_length = self.spin_output_length.value()
        min_length = self.spin_min_length.value()
        use_astar = self.cb_use_astar.isChecked()
        use_momentum = self.cb_use_momentum.isChecked()
        
        # Disable button during generation
        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Generating...")
        self.lbl_training_status.setText("Generating text...")
        
        # Create and start generation worker
        self.generation_worker = GenerationWorker(
            self.engine,
            seed_text,
            max_length,
            temperature,
            top_p,
            repetition_penalty,
            use_astar,
            use_momentum,
            min_length
        )
        self.generation_worker.generation_progress.connect(self.on_generation_progress)
        self.generation_worker.generation_complete.connect(self.on_generation_complete)
        self.generation_worker.start()
        
        self._system_log(f"Generating text with seed: '{seed_text[:50]}...'", "INFO")
        
    def on_generation_progress(self, progress: int, message: str):
        """Handle generation progress updates"""
        self.btn_generate.setText(f"Generating... {progress}%")
        self.lbl_training_status.setText(message)
        
    def on_generation_complete(self, generated_text: str, trajectory: list, metrics: dict):
        """Handle generation completion with full metrics analysis"""
        # Re-enable button
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate Text")
        self.lbl_training_status.setText("Ready")
        
        if generated_text.startswith("Error:"):
            self._system_log(generated_text, "ERROR")
            QMessageBox.critical(self, "Generation Error", generated_text)
            return
            
        # Update output displays
        preview_length = min(500, len(generated_text))
        self.txt_output.setText(generated_text[:preview_length] + 
                              ("..." if len(generated_text) > preview_length else ""))
        self.txt_full_output.setText(generated_text)
        self.last_generated_text = generated_text
        self.last_generation_metrics = metrics
        
        # Update trajectory and visualization
        self.current_trajectory = trajectory
        self.refresh_visualization()
        
        # Update status with key metrics
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        rep_rate = metrics.get('repetition_rate', 0.0) * 100
        char_count = metrics.get('character_count', len(generated_text))
        
        status_msg = (f"Generated {char_count:,} characters | "
                     f"Coherence: {coherence:.2f} | "
                     f"Diversity: {diversity:.2f} | "
                     f"Repetition: {rep_rate:.1f}%")
        self.status_bar.showMessage(status_msg, 8000)
        
        # Log completion with metrics summary
        log_msg = (f"Generation complete: {char_count:,} chars | "
                  f"Coherence: {coherence:.2f} | "
                  f"Diversity: {diversity:.2f} | "
                  f"Time: {metrics.get('generation_time', 0.0):.2f}s")
        # Log completion with metrics summary
        log_msg = (f"Generation complete: {char_count:,} chars | "
                  f"Coherence: {coherence:.2f} | "
                  f"Diversity: {diversity:.2f} | "
                  f"Time: {metrics.get('generation_time', 0.0):.2f}s")
        self._system_log(log_msg, "SUCCESS")
        
        # Save to logs
        self.logger.log_metrics(metrics)
        self.logger.log_content(
            prompt=f"Seed: {self.txt_seed.toPlainText().strip()}",
            output=generated_text,
            metrics=metrics
        )
        self._system_log("Results saved to voxel_logs/", "INFO")
        
        # Auto-analyze if metrics indicate issues
        if coherence < 0.4 or diversity < 0.3 or rep_rate > 30.0:
            self._system_log("Quality metrics indicate potential issues. Consider analyzing the text.", "WARNING")
            # Auto-show analysis dialog after 2 seconds
            QTimer.singleShot(2000, lambda: self.analyze_text(auto_show=True))
            
    def analyze_text(self, auto_show=False):
        """Analyze the generated text with comprehensive metrics"""
        if not hasattr(self, 'last_generated_text') or not self.last_generated_text:
            QMessageBox.warning(self, "No Text", "Please generate text first.")
            return
            
        if not hasattr(self, 'last_generation_metrics') or not self.last_generation_metrics:
            # Fallback to engine analysis
            tokens = self.engine.tokenizer.tokenize(self.last_generated_text)
            analysis = self.engine.tokenizer.analyze_text(tokens)
            metrics = {
                'perplexity': analysis.perplexity,
                'coherence_score': analysis.coherence_score,
                'diversity_score': analysis.diversity_score,
                'repetition_rate': analysis.repetition_rate,
                'avg_word_length': analysis.avg_word_length,
                'sentence_complexity': analysis.sentence_complexity,
                'grammatical_correctness': analysis.grammatical_correctness,
                'topic_consistency': analysis.topic_consistency,
                'character_count': len(self.last_generated_text),
                'token_count': len(tokens)
            }
        else:
            metrics = self.last_generation_metrics
            
        # Show analysis dialog
        dialog = TextAnalysisDialog(metrics, self.last_generated_text[:1000], self)
        
        if auto_show:
            dialog.show()
        else:
            dialog.exec()
            
    def clear_output(self):
        """Clear all output displays"""
        self.txt_output.clear()
        self.txt_full_output.clear()
        self.current_trajectory = []
        self.last_generated_text = ""
        self.last_generation_metrics = {}
        self.refresh_visualization()
        self._system_log("Output cleared")

# END OF PART 4

    def refresh_visualization(self):
        """Refresh the 3D visualization with improved rendering"""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Set axis properties based on Theme
            bg_color = VoxelTheme.get_color('viz_bg')
            fg_color = VoxelTheme.get_color('viz_fg')
            
            # Canvas Background
            self.fig.patch.set_facecolor(bg_color)
            self.ax.set_facecolor(bg_color)
            
            self.ax.grid(True)
            self.ax.xaxis.set_pane_color((0,0,0,0)) # Transparent planes usually look cleaner
            self.ax.yaxis.set_pane_color((0,0,0,0))
            self.ax.zaxis.set_pane_color((0,0,0,0))
            
            # Axes and Labels
            self.ax.xaxis.line.set_color(fg_color)
            self.ax.yaxis.line.set_color(fg_color)
            self.ax.zaxis.line.set_color(fg_color)
            
            self.ax.tick_params(colors=fg_color)
            self.ax.xaxis.label.set_color(fg_color)
            self.ax.yaxis.label.set_color(fg_color)
            self.ax.zaxis.label.set_color(fg_color)
            
            self.ax.set_xlabel('X (Char 1)', color=fg_color, fontsize=10, labelpad=8)
            self.ax.set_ylabel('Y (Char 2)', color=fg_color, fontsize=10, labelpad=8)
            self.ax.set_zlabel('Z (Char 3)', color=fg_color, fontsize=10, labelpad=8)
            self.ax.set_title('3D Voxel Token Space', color=fg_color, fontsize=12, pad=10)
            
            # Get density threshold from slider
            density_percent = self.slider_density.value()
            max_density = np.max(self.engine.voxel_tensor) if np.any(self.engine.voxel_tensor) else 1.0
            threshold = max_density * (1.0 - density_percent / 100.0)
            
            # Create voxel cloud with performance optimization
            indices = np.where(self.engine.voxel_tensor > threshold)
            if len(indices[0]) > 0:
                # Limit points for performance
                max_points = 10000
                if len(indices[0]) > max_points:
                    # Weighted random sampling based on density
                    densities = self.engine.voxel_tensor[indices]
                    probabilities = densities / np.sum(densities)
                    sample_idx = np.random.choice(
                        len(indices[0]),
                        min(max_points, len(indices[0])),
                        replace=False,
                        p=probabilities
                    )
                    x = indices[0][sample_idx]
                    y = indices[1][sample_idx]
                    z = indices[2][sample_idx]
                    c = densities[sample_idx]
                else:
                    x, y, z = indices
                    c = self.engine.voxel_tensor[indices]
                    
                # Use 'viridis' for high contrast on white background
                sc = self.ax.scatter(x, y, z, c=c, cmap='viridis', 
                                   marker='s', alpha=0.6, s=20, label="Voxel Density")
                
                # Add colorbar if not already present or needs update
                # Note: clearing axis removed colorbar associated with scatter? 
                # Matplotlib colorbars are tricky with clear(). We'll just add it.
                # To avoid duplicates, we might need to clear figure or store cbar reference.
                # For simplicity here, we assume tight_layout handles it or we accept overlay.
                # Better: Clear the specific colorbar axis if it exists.
                # Implementation detail: simplified for stability.
                
            # Plot current trajectory if available
            if self.current_trajectory and len(self.current_trajectory) > 1:
                trajectory = np.array(self.current_trajectory)
                # Plot trajectory line
                self.ax.plot(
                    trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color='#00ffff',
                    linewidth=2.5,
                    alpha=0.8,
                    label="Inference Trajectory"
                )
                
                # Plot start and end points with annotations
                start_pos = trajectory[0]
                end_pos = trajectory[-1]
                
                self.ax.scatter(
                    start_pos[0], start_pos[1], start_pos[2],
                    color='#00ff00',
                    s=150,
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=10
                )
                self.ax.text(
                    start_pos[0], start_pos[1], start_pos[2],
                    'Start', 
                    color='#00ff00',
                    fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
                )
                
                self.ax.scatter(
                    end_pos[0], end_pos[1], end_pos[2],
                    color='#ff0000',
                    s=150,
                    edgecolors='white',
                    linewidth=1.5,
                    zorder=10
                )
                self.ax.text(
                    end_pos[0], end_pos[1], end_pos[2],
                    'End', 
                    color='#ff0000',
                    fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none')
                )
                
                # Annotate key points along trajectory
                if len(trajectory) > 10:
                    step = max(1, len(trajectory) // 5)
                    for i in range(0, len(trajectory), step):
                        pos = trajectory[i]
                        self.ax.text(
                            pos[0], pos[1], pos[2],
                            f'Step {i}', 
                            color='#00aaff',
                            fontsize=8,
                            alpha=0.8
                        )
                        
            # Set optimal viewing angle based on trajectory
            if self.current_trajectory:
                trajectory = np.array(self.current_trajectory)
                self.ax.view_init(elev=20, azim=45)
            else:
                self.ax.view_init(elev=25, azim=35)
                
            # Set equal aspect ratio if possible
            try:
                self.ax.set_box_aspect([1,1,1])
            except:
                pass
                
            # Create legend only if there are artists with labels
            if self.ax.get_legend_handles_labels()[0]:
                legend = self.ax.legend(
                    loc='upper right', 
                    fontsize='small', 
                    facecolor='#1a1a1a',
                    edgecolor='#333333'
                )
                for text in legend.get_texts():
                    text.set_color('white')
                    
            # Adjust layout and redraw
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            error_msg = f"Visualization error: {str(e)}"
            self._system_log(error_msg, "ERROR")
            
    def show_concepts_dialog(self):
        """Display discovered concepts in a dialog"""
        try:
            concepts = self.engine.identify_concepts(top_n=10)
            if not concepts:
                return
                
            dialog = QDialog(self)
            dialog.setWindowTitle("Discovered Concept Clouds")
            dialog.resize(500, 400)
            layout = QVBoxLayout(dialog)
            
            info_label = QLabel("The following high-density distinct regions were identified in the voxel space:")
            info_label.setWordWrap(True)
            layout.addWidget(info_label)
            
            list_widget = QListWidget()
            for i, concept in enumerate(concepts):
                pos = concept['position']
                density = concept['density']
                words = ", ".join(concept['words'])
                
                item_text = (f"Concept #{i+1} (Density: {density:.1f})\n"
                             f"Position: {pos}\n"
                             f"Key Terms: {words}")
                item = QListWidgetItem(item_text)
                list_widget.addItem(item)
                
            layout.addWidget(list_widget)
            
            btn_close = QPushButton("Close")
            btn_close.clicked.connect(dialog.accept)
            layout.addWidget(btn_close)
            
            dialog.exec()
            self._system_log("Concept clouds analysis displayed")
            
        except Exception as e:
            self._system_log(f"Error showing concepts: {e}", "ERROR")
            
    def clear_visualization(self):
        """Clear the visualization and trajectory"""
        self.current_trajectory = []
        self.refresh_visualization()
        self._system_log("Visualization cleared")
        
    def refresh_model_list(self):
        """Refresh the list of saved models with detailed information"""
        self.list_models.clear()
        self.loaded_models = []
        if not os.path.exists(self.engine.model_dir):
            os.makedirs(self.engine.model_dir, exist_ok=True)
            
        # Scan for model directories
        model_dirs = []
        for item in os.listdir(self.engine.model_dir):
            item_path = os.path.join(self.engine.model_dir, item)
            if os.path.isdir(item_path):
                # Check if it's a valid model
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    model_dirs.append((item, item_path))
                    
        # Sort by modification time (newest first)
        model_dirs.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        
        # Populate list
        for model_name, model_path in model_dirs:
            try:
                with open(os.path.join(model_path, "metadata.json"), 'r') as f:
                    metadata = json.load(f)
                    
                # Create list item with rich text
                item = QListWidgetItem()
                
                # Format display text
                token_count = metadata.get('tokens_processed', 0)
                ngrams_total = sum(metadata.get('ngrams_learned', {}).values())
                memory_mb = metadata.get('memory_usage_mb', 0)
                save_time = metadata.get('save_time', 'Unknown')
                
                # Parse ISO format time
                try:
                    save_dt = datetime.datetime.fromisoformat(save_time)
                    time_str = save_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_str = save_time[:16] if len(save_time) > 16 else save_time
                    
                # Format text for display (PlainText for QListWidget)
                display_text = (
                    f"{model_name}\n"
                    f"Tokens: {token_count:,} | N-Grams: {ngrams_total:,} | Memory: {memory_mb:.1f} MB\n"
                    f"Saved: {time_str}"
                )
                
                item.setData(Qt.ItemDataRole.UserRole, model_path)
                item.setData(Qt.ItemDataRole.DisplayRole, display_text)
                item.setSizeHint(QSize(0, 60))  # Set item height
                
                self.list_models.addItem(item)
                self.loaded_models.append(model_path)
                
            except Exception as e:
                continue
                
        self._system_log(f"Found {self.list_models.count()} saved models")
        self.list_models.setStyleSheet("""
        QListWidget {
            background-color: #1a1a2a;
            border: 1px solid #333344;
            font-family: 'Consolas';
        }
        QListWidget::item {
            border-bottom: 1px solid #2a2a3a;
            padding: 5px;
        }
        QListWidget::item:selected {
            background-color: #2a3a5a;
            color: white;
        }
        """)
        
    def load_selected_model(self):
        """Load the selected model with progress feedback"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to load.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "Error", "Model directory not found.")
            return
            
        # Show loading dialog
        loading_dialog = QDialog(self)
        loading_dialog.setWindowTitle("Loading Model")
        loading_dialog.setFixedSize(300, 100)
        loading_layout = QVBoxLayout(loading_dialog)
        
        progress_label = QLabel("Loading model components...")
        progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_layout.addWidget(progress_label)
        
        progress_bar = QProgressBar()
        progress_bar.setRange(0, 100)
        loading_layout.addWidget(progress_bar)
        
        loading_dialog.show()
        QApplication.processEvents()
        
        try:
            # Load model with progress updates
            start_time = time.time()
            
            # Update progress
            progress_bar.setValue(20)
            progress_label.setText("Loading N-Gram memory...")
            QApplication.processEvents()
            
            # Load model
            success = self.engine.load_model(model_path)
            
            if success:
                # Update progress
                progress_bar.setValue(60)
                progress_label.setText("Updating visualizations...")
                QApplication.processEvents()
                
                # Refresh UI
                self.refresh_visualization()
                self.update_training_statistics({
                    'tokens_processed': self.engine.tokens_processed,
                    'ngrams_learned': self.engine.ngrams_learned,
                    'memory_usage_mb': self.engine.memory_usage_mb
                })
                self.refresh_model_list()
                
                # Update model info
                metadata_path = os.path.join(model_path, "metadata.json")
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                info_text = (
                    f"<b>Model:</b> {os.path.basename(model_path)}<br>"
                    f"<b>Tokens Processed:</b> {metadata['tokens_processed']:,}<br>"
                    f"<b>Total N-Grams:</b> {sum(metadata['ngrams_learned'].values()):,}<br>"
                    f"<b>Memory Usage:</b> {metadata.get('memory_usage_mb', 0):.1f} MB<br>"
                    f"<b>Training Time:</b> {metadata.get('training_time', 0):.1f} seconds<br>"
                    f"<b>Saved:</b> {metadata['save_time']}<br>"
                    f"<b>Version:</b> {metadata.get('model_version', 'unknown')}"
                )
                
                self.txt_model_info.setHtml(info_text)
                
                # Final progress update
                progress_bar.setValue(100)
                progress_label.setText("Model loaded successfully!")
                QApplication.processEvents()
                
                # Keep dialog visible for 0.5 seconds
                QTimer.singleShot(500, loading_dialog.accept)
                
                elapsed = time.time() - start_time
                self._system_log(f"Model loaded: {os.path.basename(model_path)} in {elapsed:.2f}s", "SUCCESS")
                QMessageBox.information(self, "Load Complete", 
                                      f"Model loaded successfully:\n{os.path.basename(model_path)}")
            else:
                loading_dialog.close()
                self._system_log("Failed to load model", "ERROR")
                
        except Exception as e:
            loading_dialog.close()
            error_msg = f"Failed to load model:\n{str(e)}\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Load Error", error_msg)
            self._system_log(f"Load error: {str(e)}", "ERROR")

# END OF PART 5

    def delete_selected_model(self):
        """Delete the selected model with confirmation"""
        selected_items = self.list_models.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a model to delete.")
            return
            
        model_path = selected_items[0].data(Qt.ItemDataRole.UserRole)
        model_name = os.path.basename(model_path)
        
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Are you sure you want to permanently delete this model?\n\n{model_name}\n\n"
            f"This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Confirm deletion with additional safeguard
                confirm_text, ok = QInputDialog.getText(
                    self,
                    "Confirm Deletion",
                    f"Type the model name to confirm deletion:\n{model_name}",
                    QLineEdit.EchoMode.Normal,
                    ""
                )
                
                if ok and confirm_text.strip() == model_name:
                    # Remove directory and contents
                    import shutil
                    shutil.rmtree(model_path)
                    
                    # Refresh list
                    self.refresh_model_list()
                    
                    self._system_log(f"Model deleted: {model_name}", "SUCCESS")
                    QMessageBox.information(self, "Deletion Complete", 
                                          f"Model successfully deleted:\n{model_name}")
                else:
                    self._system_log("Model deletion canceled - name mismatch", "WARNING")
                    
            except Exception as e:
                error_msg = f"Failed to delete model:\n{str(e)}"
                QMessageBox.critical(self, "Delete Error", error_msg)
                self._system_log(f"Delete error: {str(e)}", "ERROR")
                
    def export_model(self):
        """Export current model to selected directory"""
        if not self.engine.tokens_processed:
            QMessageBox.warning(self, "No Model", "No trained model to export. Please train first.")
            return
            
        save_path = QFileDialog.getExistingDirectory(
            self,
            "Select Export Directory",
            os.path.join(os.getcwd(), "exported_models")
        )
        
        if save_path:
            try:
                # Create export directory if needed
                export_dir = os.path.join(save_path, f"voxelpath_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(export_dir, exist_ok=True)
                
                # Show progress dialog
                progress_dialog = QDialog(self)
                progress_dialog.setWindowTitle("Exporting Model")
                progress_dialog.setFixedSize(400, 120)
                layout = QVBoxLayout(progress_dialog)
                
                status_label = QLabel("Starting export...")
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                
                layout.addWidget(status_label)
                layout.addWidget(progress_bar)
                
                progress_dialog.show()
                QApplication.processEvents()
                
                # Export components with progress updates
                steps = [
                    ("Saving N-Gram memory", 30),
                    ("Saving voxel tensor", 20),
                    ("Saving spatial graph", 20),
                    ("Saving metadata", 30)
                ]
                
                total_progress = 0
                for step_name, step_weight in steps:
                    status_label.setText(step_name)
                    QApplication.processEvents()
                    
                    # Simulate progress within step
                    for i in range(1, 11):
                        progress_bar.setValue(total_progress + int(step_weight * i / 10))
                        QApplication.processEvents()
                        time.sleep(0.05)
                        
                    total_progress += step_weight
                    
                # Actually save the model
                export_path = self.engine.save_model(os.path.basename(export_dir))
                
                # Copy to export directory
                import shutil
                if os.path.exists(export_path):
                    shutil.copytree(export_path, export_dir, dirs_exist_ok=True)
                
                # Final progress
                progress_bar.setValue(100)
                status_label.setText("Export complete!")
                QApplication.processEvents()
                
                # Keep dialog visible briefly
                QTimer.singleShot(800, progress_dialog.accept)
                
                self._system_log(f"Model exported to: {export_dir}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", 
                                      f"Model successfully exported to:\n{export_dir}")
                
            except Exception as e:
                error_msg = f"Failed to export model:\n{str(e)}"
                QMessageBox.critical(self, "Export Error", error_msg)
                self._system_log(f"Export error: {str(e)}", "ERROR")
                
    def compare_selected_models(self):
        """Compare multiple selected models with detailed metrics
        
        Note: self.compare_models_dialog is defined earlier and does the same thing.
        This method ensures the button slot works correctly.
        """
        # Reuse the existing implementation
        self.compare_models_dialog()
            
    def optimize_memory(self):
        """Run comprehensive memory optimization"""
        reply = QMessageBox.question(
            self,
            "Memory Optimization",
            "This process will prune low-frequency N-Grams and optimize the spatial graph.\n"
            "It may take several minutes for large models. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                start_time = time.time()
                self._system_log("Starting comprehensive memory optimization...", "INFO")
                
                # Show progress dialog
                progress_dialog = QDialog(self)
                progress_dialog.setWindowTitle("Optimizing Memory")
                progress_dialog.setFixedSize(400, 150)
                layout = QVBoxLayout(progress_dialog)
                
                status_label = QLabel("Analyzing memory usage...")
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                time_label = QLabel("Estimated time: Calculating...")
                
                layout.addWidget(status_label)
                layout.addWidget(progress_bar)
                layout.addWidget(time_label)
                
                progress_dialog.show()
                QApplication.processEvents()
                
                # Initial memory usage
                initial_memory = self.engine.memory_usage_mb
                
                # Optimization steps
                steps = [
                    ("Pruning infrequent N-Grams", 40),
                    ("Rebuilding spatial graph", 30),
                    ("Cleaning trajectory tunnels", 20),
                    ("Updating memory statistics", 10)
                ]
                
                total_progress = 0
                for step_name, step_weight in steps:
                    status_label.setText(step_name)
                    QApplication.processEvents()
                    
                    # Simulate progress within step
                    for i in range(1, 11):
                        progress_bar.setValue(total_progress + int(step_weight * i / 10))
                        elapsed = time.time() - start_time
                        estimated_total = elapsed / ((total_progress + step_weight * i / 10) / 100)
                        remaining = estimated_total - elapsed
                        time_label.setText(f"Elapsed: {int(elapsed)}s | Est. remaining: {int(remaining)}s")
                        QApplication.processEvents()
                        time.sleep(0.1)
                        
                    total_progress += step_weight
                    
                # Actually perform optimization
                self.engine._optimize_memory()
                
                # Final updates
                final_memory = self.engine.memory_usage_mb
                memory_saved = initial_memory - final_memory
                progress_bar.setValue(100)
                status_label.setText(f"Optimization complete! Saved {memory_saved:.1f} MB")
                time_label.setText(f"Total time: {int(time.time() - start_time)} seconds")
                QApplication.processEvents()
                
                # Keep dialog visible briefly
                QTimer.singleShot(1500, progress_dialog.accept)
                
                # Update UI
                self.refresh_visualization()
                self._update_status_bar()
                
                self._system_log(
                    f"Memory optimization complete. Saved {memory_saved:.1f} MB. "
                    f"New usage: {final_memory:.1f} MB",
                    "SUCCESS"
                )
                
            except Exception as e:
                error_msg = f"Optimization error: {str(e)}"
                QMessageBox.critical(self, "Optimization Error", error_msg)
                self._system_log(error_msg, "ERROR")

# END OF PART 6

    def save_current_model(self):
        """Save the current model state"""
        if not self.engine.tokens_processed:
            QMessageBox.warning(self, "No Model", "No trained model to save. Please train first.")
            return
            
        try:
            # Generate default name
            default_name = f"model_{len(self.engine.ngrams_learned)}_{int(time.time())}"
            name, ok = QInputDialog.getText(
                self, 
                "Save Model", 
                "Enter model name:", 
                QLineEdit.EchoMode.Normal, 
                default_name
            )
            
            if ok and name:
                # Show saving indicator in status bar
                self.status_bar.showMessage(f"Saving model '{name}'...")
                QApplication.processEvents()
                
                path = self.engine.save_model(name)
                
                self.refresh_model_list()
                self.status_bar.showMessage(f"Model saved: {name}", 5000)
                self._system_log(f"Model saved successfully: {name}", "SUCCESS")
                
        except Exception as e:
            error_msg = f"Failed to save model:\n{str(e)}"
            QMessageBox.critical(self, "Save Error", error_msg)
            self._system_log(f"Save error: {str(e)}", "ERROR")
            self.status_bar.showMessage("Save failed")
            
    def load_model_dialog(self):
        """Open file dialog to load an external model"""
        model_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Model Directory",
            self.engine.model_dir
        )
        
        if model_dir:
            if os.path.exists(os.path.join(model_dir, "metadata.json")):
                # Create progress dialog
                self.progress_dialog = QProgressDialog("Initializing model loader...", "Cancel", 0, 100, self)
                self.progress_dialog.setWindowTitle("Loading Model")
                self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
                self.progress_dialog.setAutoClose(False)
                self.progress_dialog.setAutoReset(False)
                self.progress_dialog.setValue(0)
                self.progress_dialog.show()
                
                # Start worker
                self.loader_thread = ModelLoaderWorker(self.engine, model_dir)
                self.loader_thread.progress.connect(self._on_load_progress)
                self.loader_thread.finished.connect(self._on_model_loaded)
                self.loader_thread.start()
                
                self._system_log(f"Starting async load of: {os.path.basename(model_dir)}", "INFO")
                
            else:
                QMessageBox.warning(self, "Invalid Model", "Selected directory does not contain a valid model.")

    def _on_load_progress(self, value, message):
        """Update loading progress"""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)

    def _on_model_loaded(self, success, message):
        """Handle completion of model loading"""
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
            
        if success:
            self.refresh_visualization()
            self.refresh_model_list()
            self.update_training_statistics({
                'tokens_processed': self.engine.tokens_processed,
                'ngrams_learned': self.engine.ngrams_learned,
                'memory_usage_mb': self.engine.memory_usage_mb
            })
            self._system_log(message, "SUCCESS")
            QMessageBox.information(self, "Load Complete", message)
        else:
            self._system_log(f"Load failed: {message}", "ERROR")
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n{message}")

    def export_generated_text(self):
        """Export generated text to file"""
        text = self.txt_full_output.toPlainText()
        if not text:
            QMessageBox.warning(self, "No Text", "No generated text to export.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Text",
            f"generated_text_{int(time.time())}.txt",
            "Text Files (*.txt);;Markdown Files (*.md);;All Files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self._system_log(f"Text exported to: {file_path}", "SUCCESS")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def show_trajectory_visualization(self):
        """Show detailed trajectory analysis window"""
        if not self.current_trajectory:
            QMessageBox.warning(self, "No Trajectory", "No generation trajectory available. Generate text first.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Trajectory Analysis")
        dialog.resize(900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Create figure and canvas
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0a')
        
        # Plot trajectory
        traj = np.array(self.current_trajectory)
        
        # Style axis
        ax.grid(False)
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.tick_params(colors='white')
        
        # Plot line with color gradient showing progression
        # Split into segments for colormap
        points = traj.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Calculate coherence for each step (simulated based on distance to 'center' of language)
        # In reality we would use step-by-step metrics if recorded
        
        # Simple line plot for now
        ax.plot(traj[:,0], traj[:,1], traj[:,2], c='cyan', linewidth=2, label='Trajectory')
        
        # Start/End
        ax.scatter(traj[0,0], traj[0,1], traj[0,2], c='lime', s=100, label='Start')
        ax.scatter(traj[-1,0], traj[-1,1], traj[-1,2], c='red', s=100, label='End')
        
        ax.legend(facecolor='#1a1a1a', labelcolor='white')
        fig.tight_layout()
        
        layout.addWidget(canvas)
        
        # Stats panel
        stats_layout = QHBoxLayout()
        
        # Calculate stats
        total_dist = self._calculate_trajectory_distance(traj)
        avg_step = self._calculate_avg_step_size(traj)
        displacement = np.linalg.norm(traj[-1] - traj[0])
        efficiency = displacement / total_dist if total_dist > 0 else 0
        
        stats_text = (
            f"<b>Total Distance:</b> {total_dist:.2f}<br>"
            f"<b>Avg Step Size:</b> {avg_step:.2f}<br>"
            f"<b>Net Displacement:</b> {displacement:.2f}<br>"
            f"<b>Path Efficiency:</b> {efficiency:.2%}"
        )
        
        prob_text = (
            f"<b>Steps:</b> {len(traj)}<br>"
            f"<b>Dimensionality:</b> 3D<br>"
            f"<b>Manifold:</b> Voxel Space"
        )
        
        lbl_stats = QLabel(stats_text)
        lbl_stats.setStyleSheet("color: white; font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")
        
        lbl_probs = QLabel(prob_text)
        lbl_probs.setStyleSheet("color: white; font-size: 14px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;")
        
        stats_layout.addWidget(lbl_stats)
        stats_layout.addWidget(lbl_probs)
        
        layout.addLayout(stats_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_export = QPushButton("Export Data")
        btn_export.clicked.connect(lambda: self.export_trajectory_data(traj))
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_close)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
        
    def _calculate_trajectory_distance(self, trajectory):
        """Calculate total Euclidean distance of a trajectory"""
        dist = 0
        for i in range(len(trajectory) - 1):
            dist += np.linalg.norm(trajectory[i+1] - trajectory[i])
        return dist
        
    def _calculate_avg_step_size(self, trajectory):
        """Calculate average step size"""
        if len(trajectory) < 2:
            return 0
        return self._calculate_trajectory_distance(trajectory) / (len(trajectory) - 1)
        
    def export_trajectory_data(self, trajectory):
        """Export trajectory coordinates to CSV"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Trajectory",
            f"trajectory_{int(time.time())}.csv",
            "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Step', 'X', 'Y', 'Z'])
                    for i, point in enumerate(trajectory):
                        writer.writerow([i, point[0], point[1], point[2]])
                QMessageBox.information(self, "Export Complete", f"Trajectory saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

# END OF PART 7

    def open_batch_generation(self):
        """Open batch generation dialog"""
        if not hasattr(self, 'current_corpus') or not self.current_corpus:
            QMessageBox.warning(self, "No Corpus", "Please load a training corpus first.")
            return
            
        dialog = BatchGenerationDialog(self.engine, self)
        dialog.exec()
        
    def show_about_dialog(self):
        """Show improved about dialog"""
        dialog = AboutDialog(self)
        dialog.exec()
        
    def open_documentation(self):
        """Show in-app documentation"""
        dialog = DocumentationDialog(self)
        dialog.exec()
        
    def show_tutorial(self):
        """Show interactive tutorial wizard"""
        tutorial = InteractiveTutorial(self, self)
        tutorial.show() # Non-modal to allow interaction with main window

    def closeEvent(self, event):
        """Handle application closure"""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Training in Progress",
                "Training is currently active. Do you want to stop training and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.training_worker.stop()
                self.training_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _create_knowledge_tab(self) -> QWidget:
        """Create tab for RAG Knowledge Base management"""
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = QLabel("Knowledge Base Management")
        header.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {VoxelTheme.get_color('text_primary')};")
        layout.addWidget(header)
        
        desc = QLabel("Ingest external documents to augment the model's knowledge (RAG).")
        desc.setStyleSheet(f"color: {VoxelTheme.get_color('text_secondary')}; font-size: 14px;")
        layout.addWidget(desc)
        
        # Controls Container
        controls = QFrame()
        controls.setStyleSheet(f"""
            QFrame {{
                background-color: {VoxelTheme.get_color('bg_panel')};
                border-radius: 10px;
                border: 1px solid {VoxelTheme.get_color('border')};
            }}
        """)
        control_layout = QVBoxLayout(controls)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(20, 20, 20, 20)
        
        # Ingest Button
        self.btn_ingest_doc = QPushButton("Ingest Text Document")
        self.btn_ingest_doc.setMinimumHeight(50)
        self.btn_ingest_doc.setIcon(QIcon.fromTheme("document-new"))
        self.btn_ingest_doc.setStyleSheet(f"""
            QPushButton {{
                background-color: {VoxelTheme.get_color('accent_primary')};
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton:hover {{
                background-color: {VoxelTheme.get_color('accent_secondary')};
            }}
        """)
        self.btn_ingest_doc.clicked.connect(self.ingest_document_rag)
        control_layout.addWidget(self.btn_ingest_doc)
        
        # Status
        status_layout = QHBoxLayout()
        status_label = QLabel("Current Status:")
        status_label.setStyleSheet("font-weight: bold;")
        self.lbl_rag_status = QLabel("Ready")
        self.lbl_rag_status.setStyleSheet("color: #10b981;") # Emerald green
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.lbl_rag_status)
        status_layout.addStretch()
        control_layout.addLayout(status_layout)
        
        layout.addWidget(controls)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab

    def _create_server_tab(self) -> QWidget:
        """Create Server Management Tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Status Panel
        status_group = QGroupBox("Server Status")
        status_layout = QVBoxLayout()
        
        self.server_status_label = QLabel("Status: STOPPED")
        self.server_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
        
        self.server_url_link = QLabel('<a href="http://localhost:8000/docs">Open API Documentation (Swagger UI)</a>')
        self.server_url_link.setOpenExternalLinks(True)
        self.server_url_link.setVisible(False)
        
        status_layout.addWidget(self.server_status_label)
        status_layout.addWidget(self.server_url_link)
        status_group.setLayout(status_layout)
        
        # Controls
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout()
        
        self.btn_start_server = QPushButton("Start Server")
        self.btn_start_server.setStyleSheet("background-color: #2ea043; color: white; padding: 8px;")
        self.btn_start_server.clicked.connect(self.start_server)
        
        self.btn_stop_server = QPushButton("Stop Server")
        self.btn_stop_server.setStyleSheet("background-color: #d73a49; color: white; padding: 8px;")
        self.btn_stop_server.setEnabled(False)
        self.btn_stop_server.clicked.connect(self.stop_server)
        
        controls_layout.addWidget(self.btn_start_server)
        controls_layout.addWidget(self.btn_stop_server)
        controls_group.setLayout(controls_layout)
        
        # Logs
        log_group = QGroupBox("Server Logs")
        log_layout = QVBoxLayout()
        self.server_log_view = QTextEdit()
        self.server_log_view.setReadOnly(True)
        self.server_log_view.setStyleSheet("background-color: #0d1117; color: #c9d1d9; font-family: Consolas;")
        log_layout.addWidget(self.server_log_view)
        log_group.setLayout(log_layout)
        
        layout.addWidget(status_group)
        layout.addWidget(controls_group)
        layout.addWidget(log_group)
        
        tab.setLayout(layout)
        
        # Initialize Worker
        self.server_worker = ServerWorker()
        self.server_worker.log_signal.connect(self.log_server_message)
        self.server_worker.status_signal.connect(self.update_server_status)
        
        return tab
        
    def start_server(self):
        self.server_log_view.clear()
        self.server_worker.start()
        self.btn_start_server.setEnabled(False)
        self.btn_stop_server.setEnabled(True)
        
    def stop_server(self):
        self.server_worker.stop()
        self.btn_start_server.setEnabled(True)
        self.btn_stop_server.setEnabled(False)
        
    def log_server_message(self, msg):
        self.server_log_view.append(msg)
        
    def update_server_status(self, is_running):
        if is_running:
            self.server_status_label.setText("Status: RUNNING")
            self.server_status_label.setStyleSheet("color: green; font-weight: bold; font-size: 14px;")
            self.server_url_link.setVisible(True)
            self.btn_start_server.setEnabled(False)
            self.btn_stop_server.setEnabled(True)
        else:
            self.server_status_label.setText("Status: STOPPED")
            self.server_status_label.setStyleSheet("color: red; font-weight: bold; font-size: 14px;")
            self.server_url_link.setVisible(False)
            self.btn_start_server.setEnabled(True)
            self.btn_stop_server.setEnabled(False)

    def ingest_document_rag(self):
        """Handle document ingestion for RAG asynchronously"""
        fname, _ = QFileDialog.getOpenFileName(self, "Open Document", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    text = f.read()
                    
                # Create progress dialog
                self.rag_progress = QProgressDialog("Ingesting document (generating embeddings)...", "Cancel", 0, 0, self)
                self.rag_progress.setWindowTitle("RAG Ingestion")
                self.rag_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                self.rag_progress.setAutoClose(False)
                self.rag_progress.show()
                
                # Start worker
                self.rag_worker = RagIngestionWorker(self.engine, text, os.path.basename(fname))
                self.rag_worker.finished.connect(self._on_rag_ingest_finished)
                self.rag_worker.start()
                
                self._system_log(f"Starting async ingestion of: {os.path.basename(fname)}", "INFO")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file:\n{str(e)}")

    def _on_rag_ingest_finished(self, success, message):
        """Handle completion of RAG ingestion"""
        if hasattr(self, 'rag_progress') and self.rag_progress:
            self.rag_progress.close()
            
        if success:
            self.lbl_rag_status.setText("Ready")
            self._system_log(message, "SUCCESS")
            QMessageBox.information(self, "Ingestion Complete", message)
        else:
            self.lbl_rag_status.setText("Error")
            self._system_log(f"Ingestion failed: {message}", "ERROR")
            QMessageBox.critical(self, "Ingestion Error", f"Failed to ingest document:\n{message}")
