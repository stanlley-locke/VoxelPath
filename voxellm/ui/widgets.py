from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QAbstractTableModel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class TrainingMetricsTableModel(QAbstractTableModel):
    """Model for displaying training metrics in a table"""
    def __init__(self, metrics_history=None, parent=None):
        super().__init__(parent)
        self.metrics_history = metrics_history or []
        self.headers = [
            'Batch', 'Tokens', '2-grams', '3-grams', '4-grams', '5-grams', 
            'Perplexity', 'Coherence', 'Diversity', 'Memory (MB)', 'Rate (tok/s)'
        ]
        
    def rowCount(self, parent=None):
        return len(self.metrics_history)
        
    def columnCount(self, parent=None):
        return len(self.headers)
        
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
            
        row = index.row()
        col = index.column()
        metrics = self.metrics_history[row]
        
        try:
            if col == 0:  # Batch
                return str(metrics.get('batch_idx', row) + 1)
            elif col == 1:  # Tokens
                return f"{metrics.get('tokens_processed', 0):,}"
            elif col == 2:  # 2-grams
                return f"{metrics.get('ngrams_learned', {}).get(2, 0):,}"
            elif col == 3:  # 3-grams
                return f"{metrics.get('ngrams_learned', {}).get(3, 0):,}"
            elif col == 4:  # 4-grams
                return f"{metrics.get('ngrams_learned', {}).get(4, 0):,}"
            elif col == 5:  # 5-grams
                return f"{metrics.get('ngrams_learned', {}).get(5, 0):,}"
            elif col == 6:  # Perplexity
                return f"{metrics.get('perplexity', float('inf')):.2f}"
            elif col == 7:  # Coherence
                return f"{metrics.get('coherence_score', 0.0):.2f}"
            elif col == 8:  # Diversity
                return f"{metrics.get('diversity_score', 0.0):.2f}"
            elif col == 9:  # Memory
                return f"{metrics.get('memory_usage_mb', 0.0):.1f}"
            elif col == 10:  # Rate
                return f"{metrics.get('training_rate', 0.0):.0f}"
        except (KeyError, TypeError, ValueError):
            return "N/A"
            
        return None
        
    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.headers[section]
        return super().headerData(section, orientation, role)
        
class TrainingProgressVisualizer(QWidget):
    """Widget for visualizing training progress with matplotlib"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Create figure and axes
        self.fig = Figure(figsize=(10, 6), facecolor='#2a2a3a')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        # Create subplots
        self.ax1 = self.fig.add_subplot(221, facecolor='#2a2a3a')  # Perplexity
        self.ax2 = self.fig.add_subplot(222, facecolor='#2a2a3a')  # N-Grams
        self.ax3 = self.fig.add_subplot(223, facecolor='#2a2a3a')  # Memory/Rate
        self.ax4 = self.fig.add_subplot(224, facecolor='#2a2a3a')  # Coherence/Diversity
        
        # Style the plots
        self._style_plots()
        
        # Data storage
        self.metrics_history = []
        
    def _style_plots(self):
        """Apply consistent styling to all plots"""
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.grid(True, alpha=0.3, color='#555555')
            ax.tick_params(colors='#cccccc')
            for spine in ax.spines.values():
                spine.set_color('#555555')
            ax.xaxis.label.set_color('#cccccc')
            ax.yaxis.label.set_color('#cccccc')
            ax.title.set_color('#cccccc')
            
    def update_metrics(self, metrics_history):
        """Update visualizations with new metrics data"""
        self.metrics_history = metrics_history
        self._update_plots()
        
    def _update_plots(self):
        """Update all plot data"""
        if not self.metrics_history:
            return
            
        # Extract data
        batches = [m.get('batch_idx', i) for i, m in enumerate(self.metrics_history)]
        perplexities = [m.get('perplexity', float('inf')) for m in self.metrics_history]
        coherence_scores = [m.get('coherence_score', 0.0) for m in self.metrics_history]
        diversity_scores = [m.get('diversity_score', 0.0) for m in self.metrics_history]
        memory_usage = [m.get('memory_usage_mb', 0.0) for m in self.metrics_history]
        training_rates = [m.get('training_rate', 0.0) for m in self.metrics_history]
        
        # N-Gram counts
        bigrams = [m.get('ngrams_learned', {}).get(2, 0) for m in self.metrics_history]
        trigrams = [m.get('ngrams_learned', {}).get(3, 0) for m in self.metrics_history]
        quadgrams = [m.get('ngrams_learned', {}).get(4, 0) for m in self.metrics_history]
        quintgrams = [m.get('ngrams_learned', {}).get(5, 0) for m in self.metrics_history]
        
        # Clear plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Plot 1: Perplexity
        self.ax1.plot(batches, perplexities, 'r-', linewidth=2, label='Perplexity')
        self.ax1.set_title('Model Perplexity', fontsize=10)
        self.ax1.set_xlabel('Batch')
        self.ax1.set_ylabel('Perplexity')
        self.ax1.legend(fontsize=8)
        
        # Plot 2: N-Gram counts
        self.ax2.plot(batches, bigrams, 'b-', linewidth=1, label='2-grams')
        self.ax2.plot(batches, trigrams, 'g-', linewidth=1, label='3-grams')
        self.ax2.plot(batches, quadgrams, 'y-', linewidth=1, label='4-grams')
        self.ax2.plot(batches, quintgrams, 'm-', linewidth=1, label='5-grams')
        self.ax2.set_title('N-Gram Learning Progress', fontsize=10)
        self.ax2.set_xlabel('Batch')
        self.ax2.set_ylabel('Count')
        self.ax2.legend(fontsize=6)
        
        # Plot 3: Memory and Training Rate
        ax3b = self.ax3.twinx()
        self.ax3.plot(batches, memory_usage, 'c-', linewidth=2, label='Memory')
        ax3b.plot(batches, training_rates, 'y-', linewidth=2, label='Rate')
        self.ax3.set_title('Resource Usage', fontsize=10)
        self.ax3.set_xlabel('Batch')
        self.ax3.set_ylabel('Memory (MB)', color='c')
        ax3b.set_ylabel('Tokens/sec', color='y')
        self.ax3.tick_params(axis='y', colors='c')
        ax3b.tick_params(axis='y', colors='y')
        
        # Plot 4: Quality Metrics
        self.ax4.plot(batches, coherence_scores, 'g-', linewidth=2, label='Coherence')
        self.ax4.plot(batches, diversity_scores, 'b-', linewidth=2, label='Diversity')
        self.ax4.set_title('Text Quality Metrics', fontsize=10)
        self.ax4.set_xlabel('Batch')
        self.ax4.set_ylabel('Score')
        self.ax4.legend(fontsize=8)
        
        # Apply styling
        self._style_plots()
        
        # Adjust layout
        self.fig.tight_layout()
        
        # Redraw
        self.canvas.draw()
        
    def clear(self):
        """Clear all plots"""
        self.metrics_history = []
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        self.canvas.draw()
