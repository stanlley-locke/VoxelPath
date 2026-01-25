import os
import random
import datetime
import json
import csv
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTabWidget, QWidget, QTableWidget,
    QHeaderView, QTableWidgetItem, QHBoxLayout, QProgressBar, QTextEdit,
    QDialogButtonBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QFileDialog, QMessageBox,
    QApplication
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt

from ..core.engine import VoxelPathProductionEngine

class TextAnalysisDialog(QDialog):
    """Dialog for detailed text analysis with advanced metrics"""
    def __init__(self, metrics: dict, text_sample: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Text Analysis Report")
        self.setMinimumSize(700, 600)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Comprehensive Text Quality Analysis")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tab widget for different analysis sections
        tab_widget = QTabWidget()
        
        # Tab 1: Quality Metrics
        metrics_tab = QWidget()
        metrics_layout = QVBoxLayout(metrics_tab)
        
        # Metrics table
        table = QTableWidget(10, 2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        metrics_data = [
            ("Perplexity", f"{metrics.get('perplexity', float('inf')):.2f}"),
            ("Coherence Score", f"{metrics.get('coherence_score', 0.0):.2f}"),
            ("Diversity Score", f"{metrics.get('diversity_score', 0.0):.2f}"),
            ("Flesch Reading Ease", f"{metrics.get('flesch_reading_ease', 0.0):.2f}"),
            ("Sentiment Score", f"{metrics.get('sentiment', 0.0):.2f}"),
            ("Topic Consistency", f"{metrics.get('topic_consistency', 0.0):.2f}"),
            ("Grammatical Correctness", f"{metrics.get('grammatical_correctness', 0.0):.2f}"),
            ("Repetition Rate", f"{metrics.get('repetition_rate', 0.0):.2f}"),
            ("Avg Word Length", f"{metrics.get('avg_word_length', 0.0):.2f}"),
            ("Sentence Complexity", f"{metrics.get('sentence_complexity', 0.0):.2f}"),
            ("Generation Time", f"{metrics.get('generation_time', 0.0):.2f} seconds"),
            ("Character Count", f"{metrics.get('character_count', 0)}")
        ]
        
        # Resize table to fit new rows
        table.setRowCount(len(metrics_data))
        
        for i, (metric, value) in enumerate(metrics_data):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(value))
            
        metrics_layout.addWidget(table)
        
        # Quality gauge visualization
        gauge_layout = QHBoxLayout()
        
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        grammar = metrics.get('grammatical_correctness', 0.0)
        
        # Create simple gauge visualizations
        def create_gauge(label, value, color):
            container = QVBoxLayout()
            lbl = QLabel(f"{label}: {value:.2f}")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container.addWidget(lbl)
            
            # Gauge bar
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(int(value * 100))
            
            if value < 0.3:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #ff4444; }")
            elif value < 0.7:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #ffaa00; }")
            else:
                bar.setStyleSheet("QProgressBar::chunk { background-color: #44ff44; }")
                
            container.addWidget(bar)
            return container
            
        gauge_layout.addLayout(create_gauge("Coherence", coherence, "#44ff44"))
        gauge_layout.addLayout(create_gauge("Diversity", diversity, "#4444ff"))
        gauge_layout.addLayout(create_gauge("Grammar", grammar, "#ff4444"))
        
        metrics_layout.addLayout(gauge_layout)
        
        tab_widget.addTab(metrics_tab, "Quality Metrics")
        
        # Tab 2: Text Sample
        sample_tab = QWidget()
        sample_layout = QVBoxLayout(sample_tab)
        
        sample_label = QLabel("Generated Text Sample:")
        sample_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        sample_layout.addWidget(sample_label)
        
        sample_text = QTextEdit()
        sample_text.setReadOnly(True)
        sample_text.setFont(QFont("Georgia", 11))
        sample_text.setStyleSheet("""
        QTextEdit {
            background-color: #1e1e2e;
            color: #f0f0f0;
            border: 2px solid #333344;
            padding: 10px;
            line-height: 1.4;
        }
        """)
        sample_text.setPlainText(text_sample)
        sample_layout.addWidget(sample_text)
        
        tab_widget.addTab(sample_tab, "Text Sample")
        
        # Tab 3: Recommendations
        recommendations_tab = QWidget()
        rec_layout = QVBoxLayout(recommendations_tab)
        
        recommendations = self._generate_recommendations(metrics)
        
        rec_label = QLabel("Optimization Recommendations:")
        rec_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        rec_layout.addWidget(rec_label)
        
        rec_text = QTextEdit()
        rec_text.setReadOnly(True)
        rec_text.setFont(QFont("Consolas", 10))
        rec_text.setStyleSheet("""
        QTextEdit {
            background-color: #2a2a3a;
            color: #aaffaa;
            border: 1px solid #444444;
            padding: 10px;
        }
        """)
        rec_text.setPlainText(recommendations)
        rec_layout.addWidget(rec_text)
        
        tab_widget.addTab(recommendations_tab, "Recommendations")
        
        layout.addWidget(tab_widget)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def _generate_recommendations(self, metrics: dict) -> str:
        """Generate optimization recommendations based on metrics"""
        recs = []
        recs.append("MODEL OPTIMIZATION RECOMMENDATIONS")
        recs.append("=" * 50)
        recs.append("")
        
        # Analyze metrics
        coherence = metrics.get('coherence_score', 0.0)
        diversity = metrics.get('diversity_score', 0.0)
        grammar = metrics.get('grammatical_correctness', 0.0)
        repetition = metrics.get('repetition_rate', 0.0)
        perplexity = metrics.get('perplexity', float('inf'))
        
        # Coherence recommendations
        if coherence < 0.4:
            recs.append(f"LOW COHERENCE ({coherence:.2f}):")
            recs.append("- Increase context window size")
            recs.append("- Lower temperature (0.6-0.7)")
            recs.append("- Increase repetition penalty (1.3-1.5)")
            recs.append("- Train on more coherent text sources")
            recs.append("")
            
        # Diversity recommendations
        if diversity < 0.3:
            recs.append(f"LOW DIVERSITY ({diversity:.2f}):")
            recs.append("- Increase temperature (0.9-1.2)")
            recs.append("- Decrease top-p threshold (0.7-0.8)")
            recs.append("- Add more varied training data")
            recs.append("- Reduce repetition penalty (0.9-1.1)")
            recs.append("")
            
        # Grammar recommendations
        if grammar < 0.6:
            recs.append(f"POOR GRAMMAR ({grammar:.2f}):")
            recs.append("- Train on more grammatically correct text")
            recs.append("- Increase bigram/trigram weight")
            recs.append("- Add explicit grammar rules")
            recs.append("- Use lower temperature (0.5-0.7)")
            recs.append("")
            
        # Repetition recommendations
        if repetition > 0.3:
            recs.append(f"HIGH REPETITION ({repetition:.2f}):")
            recs.append("- Increase repetition penalty (1.4-2.0)")
            recs.append("- Increase temperature (0.9-1.1)")
            recs.append("- Use more aggressive pruning")
            recs.append("")
            
        # Perplexity recommendations
        if perplexity > 100:
            recs.append(f"HIGH PERPLEXITY ({perplexity:.2f}):")
            recs.append("- More training data needed")
            recs.append("- Longer training time")
            recs.append("- Check data quality and preprocessing")
            recs.append("")
            
        # General recommendations
        recs.append("GENERAL OPTIMIZATION STRATEGIES:")
        recs.append("- Use validation set for early stopping")
        recs.append("- Monitor metrics over multiple generations")
        recs.append("- Experiment with different context lengths")
        recs.append("- Try different sampling strategies")
        recs.append("- Consider fine-tuning on domain-specific data")
        
        if not recs[2:]:  # If no specific recommendations
            recs.append("\nEXCELLENT METRICS! No major optimizations needed.")
            recs.append("Continue monitoring and consider:")
            recs.append("- Increasing generation length")
            recs.append("- Exploring different creative modes")
            recs.append("- Testing with more challenging prompts")
            
        return "\n".join(recs)
        
class BatchGenerationDialog(QDialog):
    """Dialog for batch text generation with parameter variation"""
    def __init__(self, engine: VoxelPathProductionEngine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("Advanced Batch Text Generation")
        self.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        # Prompt Source Selection (Tabs)
        prompt_group = QGroupBox("Prompt Source")
        prompt_layout = QVBoxLayout()
        self.prompt_tabs = QTabWidget()
        
        # Tab 1: Single Seed (Existing logic)
        self.tab_single = QWidget()
        single_layout = QVBoxLayout(self.tab_single)
        
        self.example_seeds = {
            "Classic Literature": "it was the best of times, it was the worst of times",
            "Philosophical": "to be or not to be, that is the question",
            "Mystery": "the door opened slowly, revealing a room bathed in shadows",
            "Sci-Fi": "the stars stretched endlessly before the spacecraft",
            "Fantasy": "once upon a time, in a kingdom far away",
            "Custom": ""
        }
        
        self.combo_seed_examples = QComboBox()
        self.combo_seed_examples.addItems(self.example_seeds.keys())
        self.combo_seed_examples.currentTextChanged.connect(self.update_seed_text)
        single_layout.addWidget(self.combo_seed_examples)
        
        self.txt_seed = QTextEdit()
        self.txt_seed.setPlaceholderText("Enter seed text for batch generation")
        self.txt_seed.setMaximumHeight(100)
        single_layout.addWidget(self.txt_seed)
        
        self.prompt_tabs.addTab(self.tab_single, "Single Prompt")
        
        # Tab 2: Load from File
        self.tab_file = QWidget()
        file_layout = QVBoxLayout(self.tab_file)
        
        file_selection_layout = QHBoxLayout()
        self.txt_prompts_file = QLineEdit()
        self.txt_prompts_file.setPlaceholderText("Select a .txt file with one prompt per line")
        self.btn_browse_prompts = QPushButton("Browse...")
        self.btn_browse_prompts.clicked.connect(self.browse_prompts_file)
        
        file_selection_layout.addWidget(self.txt_prompts_file)
        file_selection_layout.addWidget(self.btn_browse_prompts)
        file_layout.addLayout(file_selection_layout)
        
        file_help = QLabel("File should contain one prompt per line. Each prompt will be processed.")
        file_help.setStyleSheet("color: #888; font-style: italic;")
        file_layout.addWidget(file_help)
        file_layout.addStretch()
        
        self.prompt_tabs.addTab(self.tab_file, "Load from File")
        
        prompt_layout.addWidget(self.prompt_tabs)
        prompt_group.setLayout(prompt_layout)
        layout.addWidget(prompt_group)
        
        # Parameters with ranges
        param_group = QGroupBox("Generation Parameters (range for variation)")
        param_layout = QGridLayout()
        
        # Number of texts
        param_layout.addWidget(QLabel("Variations per prompt:"), 0, 0)
        self.spin_num_texts = QSpinBox()
        self.spin_num_texts.setRange(1, 50)
        self.spin_num_texts.setValue(10)
        self.spin_num_texts.setToolTip("How many variations to generate for EACH prompt seed")
        param_layout.addWidget(self.spin_num_texts, 0, 1)
        
        # Length range
        param_layout.addWidget(QLabel("Length range:"), 1, 0)
        length_layout = QHBoxLayout()
        self.spin_min_length = QSpinBox()
        self.spin_min_length.setRange(50, 2000)
        self.spin_min_length.setValue(100)
        self.spin_max_length = QSpinBox()
        self.spin_max_length.setRange(50, 2000)
        self.spin_max_length.setValue(300)
        length_layout.addWidget(self.spin_min_length)
        length_layout.addWidget(QLabel("to"))
        length_layout.addWidget(self.spin_max_length)
        param_layout.addLayout(length_layout, 1, 1)
        
        # Temperature range
        param_layout.addWidget(QLabel("Temperature range:"), 2, 0)
        temp_layout = QHBoxLayout()
        self.spin_min_temp = QDoubleSpinBox()
        self.spin_min_temp.setRange(0.1, 2.0)
        self.spin_min_temp.setValue(0.7)
        self.spin_max_temp = QDoubleSpinBox()
        self.spin_max_temp.setRange(0.1, 2.0)
        self.spin_max_temp.setValue(1.2)
        temp_layout.addWidget(self.spin_min_temp)
        temp_layout.addWidget(QLabel("to"))
        temp_layout.addWidget(self.spin_max_temp)
        param_layout.addLayout(temp_layout, 2, 1)
        
        # Top-P range
        param_layout.addWidget(QLabel("Top-P range:"), 3, 0)
        top_p_layout = QHBoxLayout()
        self.spin_min_top_p = QDoubleSpinBox()
        self.spin_min_top_p.setRange(0.1, 1.0)
        self.spin_min_top_p.setValue(0.8)
        self.spin_max_top_p = QDoubleSpinBox()
        self.spin_max_top_p.setRange(0.1, 1.0)
        self.spin_max_top_p.setValue(0.95)
        top_p_layout.addWidget(self.spin_min_top_p)
        top_p_layout.addWidget(QLabel("to"))
        top_p_layout.addWidget(self.spin_max_top_p)
        param_layout.addLayout(top_p_layout, 3, 1)
        
        # Repetition penalty range
        param_layout.addWidget(QLabel("Repetition penalty range:"), 4, 0)
        rep_layout = QHBoxLayout()
        self.spin_min_rep = QDoubleSpinBox()
        self.spin_min_rep.setRange(0.5, 3.0)
        self.spin_min_rep.setValue(1.0)
        self.spin_max_rep = QDoubleSpinBox()
        self.spin_max_rep.setRange(0.5, 3.0)
        self.spin_max_rep.setValue(1.5)
        rep_layout.addWidget(self.spin_min_rep)
        rep_layout.addWidget(QLabel("to"))
        rep_layout.addWidget(self.spin_max_rep)
        param_layout.addLayout(rep_layout, 4, 1)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Output options
        output_group = QGroupBox("Output Options")
        output_layout = QGridLayout()
        
        output_layout.addWidget(QLabel("Output directory:"), 0, 0)
        self.txt_output_dir = QLineEdit()
        self.txt_output_dir.setText(os.path.join(os.getcwd(), "generated_texts"))
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        output_layout.addWidget(self.txt_output_dir, 0, 1)
        output_layout.addWidget(self.btn_browse, 0, 2)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Output Format
        output_layout.addWidget(QLabel("Format:"), 1, 2) # Adding to grid at col 2
        self.combo_format = QComboBox()
        self.combo_format.addItems(["Text Files (.txt)", "JSONL Dataset (.jsonl)"])
        output_layout.addWidget(self.combo_format, 1, 3) 

        # Shift naming combo to row 2
        output_layout.addWidget(QLabel("File naming:"), 2, 0)
        self.combo_naming = QComboBox()
        self.combo_naming.addItems([
            "sequential (text_001.txt)",
            "timestamp (text_20240101_123045.txt)", 
            "metrics (text_coherence_0.85.txt)"
        ])
        output_layout.addWidget(self.combo_naming, 2, 1, 1, 3)
        
        self.cb_analyze = QCheckBox("Include analysis in filename / Export Summary CSV")
        self.cb_analyze.setChecked(True)
        self.cb_analyze.setToolTip("If checked, creates batch_summary.csv in output directory")
        output_layout.addWidget(self.cb_analyze, 3, 0, 1, 4)
        
        # Progress and controls
        progress_layout = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        progress_layout.addWidget(self.progress)
        
        button_layout = QHBoxLayout()
        self.btn_generate = QPushButton("Start Batch Generation")
        self.btn_generate.clicked.connect(self.start_generation)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addWidget(self.btn_generate)
        button_layout.addWidget(self.btn_cancel)
        progress_layout.addLayout(button_layout)
        
        layout.addLayout(progress_layout)
        
        self.setLayout(layout)
        self.generated_files = []
        
        # Set default seed
        self.update_seed_text("Classic Literature")
        
    def update_seed_text(self, example_name: str):
        """Update seed text when example is selected"""
        seed_text = self.example_seeds.get(example_name, "")
        self.txt_seed.setPlainText(seed_text)
        
    def browse_directory(self):
        """Browse for output directory"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory",
            self.txt_output_dir.text()
        )
        if directory:
            self.txt_output_dir.setText(directory)

    def browse_prompts_file(self):
        """Browse for prompts file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Prompts File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.txt_prompts_file.setText(file_path)
            
    def start_generation(self):
        """Start batch generation"""
        # 1. Gather Seeds
        seeds = []
        if self.prompt_tabs.currentIndex() == 0:  # Single
            text = self.txt_seed.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Input Error", "Please enter seed text.")
                return
            seeds.append(text)
        else:  # File
            path = self.txt_prompts_file.text().strip()
            if not os.path.exists(path):
                QMessageBox.warning(self, "File Error", "Prompts file not found.")
                return
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    seeds = [line.strip() for line in f if line.strip()]
                if not seeds:
                    QMessageBox.warning(self, "File Error", "File is empty.")
                    return
            except Exception as e:
                QMessageBox.critical(self, "Read Error", f"Error reading prompts: {e}")
                return

        output_dir = self.txt_output_dir.text()
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Directory Error", f"Could not create directory: {str(e)}")
                return
                
        # Get parameters
        variations_per_seed = self.spin_num_texts.value()
        total_generations = len(seeds) * variations_per_seed
        
        min_length = self.spin_min_length.value()
        max_length = self.spin_max_length.value()
        min_temp = self.spin_min_temp.value()
        max_temp = self.spin_max_temp.value()
        min_top_p = self.spin_min_top_p.value()
        max_top_p = self.spin_max_top_p.value()
        min_rep = self.spin_min_rep.value()
        max_rep = self.spin_max_rep.value()
        
        output_format = self.combo_format.currentText()
        is_jsonl = "JSONL" in output_format
        
        # Prepare output files
        jsonl_file = None
        if is_jsonl:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            jsonl_path = os.path.join(output_dir, f"batch_dataset_{timestamp}.jsonl")
            try:
                jsonl_file = open(jsonl_path, 'w', encoding='utf-8')
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Cannot open JSONL file: {e}")
                return

        self.btn_generate.setEnabled(False)
        self.progress.setValue(0)
        self.generated_files = [] # Store tuple (path, metrics) or dict for summary
        summary_rows = []
        
        completed_count = 0
        
        try:
            for seed_idx, seed_text in enumerate(seeds):
                for i in range(variations_per_seed):
                    if self.isHidden():  # Dialog was closed
                        break
                        
                    # Sample parameters
                    length = random.randint(min_length, max_length)
                    temperature = random.uniform(min_temp, max_temp)
                    top_p = random.uniform(min_top_p, max_top_p)
                    repetition_penalty = random.uniform(min_rep, max_rep)
                    
                    # Generate
                    generated_text, _, metrics = self.engine.generate_text(
                        seed_text=seed_text,
                        max_length=length,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        use_astar=False,
                        use_momentum=True,
                        min_length=min_length
                    )
                    
                    # Convert metrics object to dict for serialization
                    if hasattr(metrics, 'to_dict'):
                        metrics_dict = metrics.to_dict()
                    else:
                        metrics_dict = metrics # Fallback if already dict
                    
                    # Store data
                    record = {
                        "prompt": seed_text,
                        "completion": generated_text,
                        "temperature": temperature,
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "metrics": metrics_dict
                    }
                    
                    # 1. JSONL Export
                    if is_jsonl and jsonl_file:
                        jsonl_file.write(json.dumps(record) + "\n")
                        jsonl_file.flush()
                        
                    # 2. Text File Export (Only if NOT JSONL, or maybe both? Let's do Text files only if txt selected)
                    if not is_jsonl:
                        naming_scheme = self.combo_naming.currentText()
                        include_analysis = self.cb_analyze.isChecked()
                        
                        base_name = f"text_s{seed_idx}_v{i}"
                        if "sequential" in naming_scheme:
                            filename = f"{base_name}.txt"
                        elif "timestamp" in naming_scheme:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{base_name}_{ts}.txt"
                        else:  # metrics
                            coh = metrics_dict.get('coherence_score', 0.0)
                            div = metrics_dict.get('diversity_score', 0.0)
                            filename = f"{base_name}_coh{coh:.2f}_div{div:.2f}.txt"
                            
                        filepath = os.path.join(output_dir, filename)
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(f"Prompt: {seed_text}\n")
                            f.write(f"{'-'*20}\n")
                            f.write(generated_text)
                            
                        self.generated_files.append((filepath, metrics_dict))

                    # 3. Add to Summary
                    if self.cb_analyze.isChecked():
                        row = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "prompt": seed_text[:50] + "...", # Truncate for CSV
                            "temperature": f"{temperature:.2f}",
                            "coherence": f"{metrics_dict.get('coherence_score', 0):.4f}",
                            "diversity": f"{metrics_dict.get('diversity_score', 0):.4f}",
                            "flesch": f"{metrics_dict.get('flesch_reading_ease', 0):.2f}",
                            "sentiment": f"{metrics_dict.get('sentiment', 0):.2f}"
                        }
                        summary_rows.append(row)
                    
                    completed_count += 1
                    progress = int(completed_count / total_generations * 100)
                    self.progress.setValue(progress)
                    QApplication.processEvents()
                    
        finally:
            if jsonl_file:
                jsonl_file.close()

        # Write Batch Summary CSV
        if summary_rows:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(output_dir, f"batch_summary_{ts}.csv")
            try:
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ["timestamp", "prompt", "temperature", "coherence", "diversity", "flesch", "sentiment"]
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_rows)
            except Exception as e:
                print(f"Failed to write summary CSV: {e}")

        self.btn_generate.setEnabled(True)
        
        msg = f"Batch generation complete!\nProcessed {len(seeds)} prompts.\nGenerated {completed_count} texts."
        if is_jsonl:
            msg += f"\n\nSaved to JSONL dataset."
        if summary_rows:
            msg += f"\nSaved batch_summary.csv"
            
        QMessageBox.information(self, "Complete", msg)
        self.accept()
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
