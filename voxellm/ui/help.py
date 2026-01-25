
import os
import webbrowser
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextBrowser, 
    QPushButton, QTabWidget, QWidget, QListWidget, QListWidgetItem,
    QDialogButtonBox, QScrollArea, QFrame
)
from PyQt6.QtGui import QFont, QPixmap, QDesktopServices, QAction, QColor
from PyQt6.QtCore import Qt, QUrl, QSize, QTimer

from .styles import VoxelTheme

class AboutDialog(QDialog):
    """Enhanced About Dialog with tabs for detailed info"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About VoxelLM")
        self.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(self)
        
        # Header with Logo/Title
        header = QLabel("VoxelLM: Geometric Language Modeling")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"color: {VoxelTheme.get_color('accent_primary')}; margin-bottom: 10px;")
        layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_about_tab(), "About")
        tabs.addTab(self._create_architecture_tab(), "Architecture")
        tabs.addTab(self._create_credits_tab(), "Credits & License")
        layout.addWidget(tabs)
        
        # Close Button
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.accept)
        layout.addWidget(btn_box)
        
    def _create_about_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("border: none; background-color: transparent;")
        browser.setHtml(f"""
            <h3 style="color: {VoxelTheme.get_color('accent_secondary')};">Project Overview</h3>
            <p>VoxelLM is an experimental AI research platform that explores <b>Geometric Language Modeling</b>. 
            Unlike traditional Transformers that use high-dimensional vector embeddings, VoxelLM maps token 
            relationships into a structured 3D coordinate space (Voxel Grid).</p>
            
            <p>This approach treats text generation as a <b>path planning problem</b> through semantic geometry, 
            applying concepts from physics (momentum, gravity) to navigational linguistics.</p>
            
            <p><b>Version:</b> 4.0 (Production)</p>
            <p><b>Status:</b> Research Prototype</p>
        """)
        layout.addWidget(browser)
        return widget
        
    def _create_architecture_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        browser = QTextBrowser()
        browser.setStyleSheet("border: none; background-color: transparent;")
        browser.setHtml(f"""
            <h3 style="color: {VoxelTheme.get_color('accent_primary')};">Core Technologies</h3>
            <ul>
                <li><b>3D Voxel Density Mapping:</b> Tokens are assigned (x, y, z) coordinates based on semantic relationships.</li>
                <li><b>Geometric Momentum:</b> Generation follows a trajectory with simulated mass and velocity.</li>
                <li><b>Hierarchical N-Gram Memory:</b> Efficient storage of token sequences (2-gram to 5-gram).</li>
                <li><b>A* Path Planning:</b> Optional beam search variant using geometric distance heuristics.</li>
            </ul>
            
            <h4 style="color: {VoxelTheme.get_color('accent_error')};">Why Voxels?</h4>
            <p>By discretizing language space, we can visualize the "shape" of a conversation and detect 
            semantic drift as physical distance.</p>
        """)
        layout.addWidget(browser)
        return widget
        
    def _create_credits_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setStyleSheet("border: none; background-color: transparent;")
        browser.setHtml("""
            <h3>Credits</h3>
            <p><b>Developer:</b> Stanley</p>
            <p><b>Research Team:</b> VoxelPath AI</p>
            
            <h3>License</h3>
            <p>This project is licensed under the <b>MIT License</b>.</p>
            <p>Free for academic and commercial use with attribution.</p>
            
            <p><a href="https://github.com/voxelpath/voxellm" style="color: {VoxelTheme.get_color('accent_primary')};">View on GitHub</a></p>
        """)
        layout.addWidget(browser)
        return widget

class DocumentationDialog(QDialog):
    """In-app documentation browser"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VoxelLM Documentation")
        self.resize(900, 600)
        
        layout = QHBoxLayout(self)
        
        # Sidebar
        self.topic_list = QListWidget()
        self.topic_list.setFixedWidth(200)
        self.topic_list.itemClicked.connect(self._load_topic)
        layout.addWidget(self.topic_list)
        
        # Content Area
        self.content_browser = QTextBrowser()
        self.content_browser.setOpenExternalLinks(True)
        layout.addWidget(self.content_browser)
        
        # Populate Topics
        self.topics = {
            "Getting Started": self._get_start_content(),
            "Training Models": self._get_train_content(),
            "Generating Text": self._get_gen_content(),
            "Visualization": self._get_viz_content(),
            "Chat Interface": self._get_chat_content(),
            "Advanced": self._get_advanced_content()
        }
        
        for topic in self.topics:
            self.topic_list.addItem(topic)
            
        self.topic_list.setCurrentRow(0)
        self._load_topic(self.topic_list.item(0))
        
    def _load_topic(self, item):
        title = item.text()
        content = self.topics.get(title, "<h1>Content not found</h1>")
        self.content_browser.setHtml(f"<h1>{title}</h1>{content}")
        
    def _get_start_content(self):
        return """
        <p>Welcome to VoxelLM! Follow these steps to create your first AI model:</p>
        <ol>
            <li><b>Prepare Data:</b> Find a clean text file field (.txt) or JSONL dataset.</li>
            <li><b>Load Corpus:</b> Click 'Load Traning Corpus' in the Training tab.</li>
            <li><b>Train:</b> Click 'Start Training' and watch the progress bars.</li>
            <li><b>Save:</b> Once trained, go to 'Models' tab -> 'Export Model'.</li>
        </ol>
        <p><i>Tip: Start with a small file (100KB) to test quickly.</i></p>
        """

    def _get_train_content(self):
        return """
        <h3>Training Parameters</h3>
        <ul>
            <li><b>N-Gram Order:</b> Higher values (4-5) copy text more, lower values (2-3) are more creative but chaotic.</li>
            <li><b>Voxel Size:</b> Determines the resolution of the 3D semantic map.</li>
        </ul>
        <h3>Metrics</h3>
        <p><b>Perplexity:</b> Measures how 'surprised' the model is by the text. Lower is better.</p>
        """

    def _get_gen_content(self):
        return """
        <h3>Text Generation</h3>
        <p>Switch to the <b>Generation</b> tab to test your model.</p>
        <ul>
            <li><b>Seed Text:</b> The starting prompt for the AI.</li>
            <li><b>Temperature:</b> Controls randomness. High (>1.0) is chaotic, Low (<0.5) is repetitive.</li>
            <li><b>A* Pathfinding:</b> Enables geometric planning for better coherence.</li>
        </ul>
        """

    def _get_viz_content(self):
        return """
        <h3>3D Voxel Space</h3>
        <p>The visualization shows the model's memory structure.</p>
        <ul>
            <li><b>Points:</b> Represent dense clusters of related tokens.</li>
            <li><b>Lines:</b> Show the trajectory of the generated text through memory.</li>
            <li><b>Color:</b> Indicates density or semantic relevance.</li>
        </ul>
        <p>Use your mouse to rotate and zoom the 3D view.</p>
        """

    def _get_chat_content(self):
        return """
        <h3>Conversational Mode</h3>
        <p>The Chat tab allows you to talk to your model in a standard messenger interface.</p>
        <p><b>Note:</b> For best results, your model should be trained on <i>dialogue datasets</i> (like movie scripts or chat logs).</p>
        """

    def _get_advanced_content(self):
        return """
        <h3>Advanced Features</h3>
        <ul>
            <li><b>Momentum:</b> Adds physical inertia to the generation trajectory, preventing sudden topic shifts.</li>
            <li><b>Batch Generation:</b> Generate hundreds of variations at once (Tools -> Batch Gen).</li>
            <li><b>Model Comparison:</b> Select multiple models in the list to compare their stats side-by-side.</li>
        </ul>
        """

class InteractiveTutorial(QDialog):
    """Wizard-style interactive tutorial"""
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("VoxelLM Interactive Guide")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.resize(400, 200)
        self.move(100, 100) # Initial position
        
        self.steps = [
            ("Welcome", "This guide will walk you through VoxelLM's main features.\n\nClick 'Next' to begin.", None),
            ("Training Tab", "This is the Training Dashboard.\n\n1. Load a text file (.txt).\n2. Adjust N-Gram settings.\n3. Click 'Start Training' to build your model.", self._focus_training),
            ("Visualization", "The 3D View shows your model learning.\n\n- Rotate with Left Click\n- Zoom with Scroll Wheel\n- Denser clusters mean stronger memories.", self._focus_viz),
            ("Generation", "Test your model here.\n\nType a prompt in 'Seed Text' and hit Generate to see the results!", self._focus_gen),
            ("Chat Mode", "Use the Chat tab for conversational interaction.\n\nPerfect for dialogue-based models.", self._focus_chat),
            ("Done", "You're all set!\n\nCheck 'Help > Documentation' for more details.", None)
        ]
        
        self.current_step = 0
        
        layout = QVBoxLayout(self)
        
        self.lbl_title = QLabel()
        self.lbl_title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet(f"color: {VoxelTheme.get_color('accent_secondary')};")
        layout.addWidget(self.lbl_title)
        
        self.lbl_text = QLabel()
        self.lbl_text.setWordWrap(True)
        self.lbl_text.setFont(QFont("Arial", 10))
        layout.addWidget(self.lbl_text)
        
        layout.addStretch()
        
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.clicked.connect(self.prev_step)
        self.btn_next = QPushButton("Next")
        self.btn_next.clicked.connect(self.next_step)
        
        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_next)
        layout.addLayout(btn_layout)
        
        self.update_ui()
        
    def _focus_training(self):
        self.main_window.tab_widget.setCurrentIndex(0)
        
    def _focus_viz(self):
        self.main_window.tab_widget.setCurrentIndex(0) 
        # Ideally focus specifically on viz panel if possible, but tab switch is good enough
        
    def _focus_gen(self):
        self.main_window.tab_widget.setCurrentIndex(1)
        
    def _focus_chat(self):
        self.main_window.tab_widget.setCurrentIndex(2)
        
    def update_ui(self):
        title, text, action = self.steps[self.current_step]
        self.lbl_title.setText(f"{self.current_step + 1}/{len(self.steps)}: {title}")
        self.lbl_text.setText(text)
        
        self.btn_prev.setEnabled(self.current_step > 0)
        if self.current_step == len(self.steps) - 1:
            self.btn_next.setText("Finish")
        else:
            self.btn_next.setText("Next")
            
        if action:
            action()
            
    def next_step(self):
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_ui()
        else:
            self.accept()
            
    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.update_ui()
