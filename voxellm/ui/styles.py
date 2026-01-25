
from PyQt6.QtGui import QColor, QPalette

class VoxelTheme:
    # Theme configuration
    _mode = "dark" 
    
    # Palettes
    PALETTES = {
        "light": {
            "bg_dark": "#ffffff",
            "bg_panel": "#f6f8fa",
            "bg_input": "#ffffff",
            "text_primary": "#24292f",
            "text_secondary": "#57606a",
            "border": "#d0d7de",
            "accent_primary": "#0969da",
            "accent_secondary": "#2da44e",
            "accent_error": "#cf222e",
            "viz_bg": "white",
            "viz_fg": "black",
            "viz_pane": "#f0f0f0"
        },
        "dark": {
            "bg_dark": "#0d1117",
            "bg_panel": "#161b22",
            "bg_input": "#010409",
            "text_primary": "#c9d1d9",
            "text_secondary": "#8b949e",
            "border": "#30363d",
            "accent_primary": "#58a6ff",
            "accent_secondary": "#238636",
            "accent_error": "#f85149",
            "viz_bg": "#0d1117",
            "viz_fg": "white",
            "viz_pane": "#161b22"
        }
    }
    
    @classmethod
    def get_color(cls, key):
        return cls.PALETTES[cls._mode].get(key, "#ff0000")
        
    @classmethod
    def toggle(cls, app):
        new_mode = "dark" if cls._mode == "light" else "light"
        cls.apply_theme(app, new_mode)
        return new_mode

    @classmethod
    def apply_theme(cls, app, mode="dark"):
        cls._mode = mode
        colors = cls.PALETTES[mode]
        
        stylesheet = f"""
        QMainWindow {{
            background-color: {colors['bg_dark']};
        }}
        
        QWidget {{
            color: {colors['text_primary']};
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 10pt;
        }}
        
        QFrame, QGroupBox {{
            background-color: {colors['bg_panel']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
        }}
        
        QGroupBox {{
            margin-top: 24px;
            padding-top: 10px;
            font-weight: bold;
            font-size: 11pt;
            background-color: {colors['bg_dark']};
            border: 1px solid {colors['border']};
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top chat;
            padding: 0 10px;
            color: {colors['text_primary']};
            background-color: {colors['bg_dark']};
        }}
        
        QPushButton {{
            background-color: {colors['bg_panel']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 5px 16px;
            color: {colors['text_primary']};
            font-weight: 600;
        }}
        
        QPushButton:hover {{
            border-color: {colors['accent_primary']};
        }}
        
        /* Primary Action Button */
        .primary_button {{
            background-color: {colors['accent_secondary']};
            color: white;
            border: 1px solid {colors['border']};
        }}
        .primary_button:hover {{
            background-color: {colors['accent_secondary']}dd; /* Slight transparent */
        }}
        
        /* Danger Button */
        .danger_button {{
            color: {colors['accent_error']};
            border: 1px solid {colors['accent_error']};
            background-color: transparent;
        }}
        .danger_button:hover {{
            background-color: {colors['accent_error']};
            color: white;
        }}
        
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
            background-color: {colors['bg_input']};
            border: 1px solid {colors['border']};
            border-radius: 6px;
            padding: 5px 12px;
            font-family: 'Consolas', 'Monospace';
            color: {colors['text_primary']};
        }}
        
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{
            border: 2px solid {colors['accent_primary']};
            padding: 4px 11px;
        }}
        
        QTabWidget::pane {{
            border: 1px solid {colors['border']};
            background-color: {colors['bg_dark']};
            border-radius: 6px;
        }}
        
        QTabBar::tab {{
            color: {colors['text_secondary']};
            padding: 8px 16px;
            background-color: transparent;
        }}
        
        QTabBar::tab:selected {{
            color: {colors['text_primary']};
            border-bottom: 2px solid {colors['accent_error']}; /* Orange/Red accent line */
            font-weight: 600;
        }}
        
        QTableWidget {{
            background-color: {colors['bg_dark']};
            gridline-color: {colors['border']};
            border: 1px solid {colors['border']};
            alternate-background-color: {colors['bg_panel']};
        }}
        
        QHeaderView::section {{
            background-color: {colors['bg_panel']};
            padding: 8px;
            border: none;
            border-bottom: 1px solid {colors['border']};
            color: {colors['text_secondary']};
            font-weight: 600;
        }}
        
        QSlider::groove:horizontal {{
            border: 1px solid {colors['border']};
            height: 4px;
            background: {colors['bg_panel']};
        }}

        QSlider::handle:horizontal {{
            background: {colors['accent_primary']};
            border: 1px solid {colors['accent_primary']};
            width: 16px;
            height: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }}
        
        QStatusBar {{
            background-color: {colors['bg_panel']};
            color: {colors['text_secondary']};
            border-top: 1px solid {colors['border']};
        }}
        
        QMenu {{
            background-color: {colors['bg_dark']};
            color: {colors['text_primary']};
            border: 1px solid {colors['border']};
        }}
        QMenu::item:selected {{
            background-color: {colors['accent_primary']};
            color: white;
        }}

        /* Dialog Styling Fixes */
        /* Dialog Styling Fixes */
        QDialog {{
            background-color: {colors['bg_dark']};
        }}
        
        QDialog QTextBrowser {{
            background-color: transparent;
            border: none;
            color: {colors['text_primary']};
        }}
        
        QDialog QLabel {{
            background-color: transparent;
            color: {colors['text_primary']};
        }}
        
        QDialog QTabWidget::pane {{
            border: none;
            background: transparent;
        }}
        
        QDialog QTabBar::tab {{
            background: transparent;
            color: {colors['text_secondary']};
            padding: 8px 12px;
        }}
        
        QDialog QTabBar::tab:selected {{
            color: {colors['accent_primary']}; /* Use primary accent for dialog tabs */
            border-bottom: 2px solid {colors['accent_primary']};
        }}
        """
        
        app.setStyle("Fusion")
        app.setStyleSheet(stylesheet)
