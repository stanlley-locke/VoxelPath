
import os
import csv
import datetime
from typing import Dict, Any

class ResultLogger:
    """Handles logging of generation results and metrics"""
    
    def __init__(self, log_dir: str = "voxel_logs"):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            
        self.metrics_file = os.path.join(self.log_dir, "analysis_metrics.csv")
        self.content_file = os.path.join(self.log_dir, "generation_history.txt")
        
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.metrics_file):
            headers = [
                "timestamp",
                "character_count",
                "generation_time",
                "perplexity",
                "coherence_score",
                "diversity_score",
                "repetition_rate",
                "flesch_reading_ease",
                "sentiment_score",
                "grammatical_correctness",
                "topic_consistency"
            ]
            try:
                with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            except Exception as e:
                print(f"Error initializing CSV log: {e}")

    def log_metrics(self, metrics: Dict[str, Any]):
        """Append metrics to the CSV log"""
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            row = [
                timestamp,
                metrics.get("character_count", 0),
                f"{metrics.get('generation_time', 0.0):.4f}",
                f"{metrics.get('perplexity', 0.0):.4f}",
                f"{metrics.get('coherence_score', 0.0):.4f}",
                f"{metrics.get('diversity_score', 0.0):.4f}",
                f"{metrics.get('repetition_rate', 0.0):.4f}",
                f"{metrics.get('flesch_reading_ease', 0.0):.2f}",
                f"{metrics.get('sentiment', 0.0):.4f}",
                f"{metrics.get('grammatical_correctness', 0.0):.4f}",
                f"{metrics.get('topic_consistency', 0.0):.4f}"
            ]
            
            with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"Error logging metrics: {e}")

    def log_content(self, prompt: str, output: str, metrics: Dict[str, Any] = None):
        """Append prompt and generated output to text log"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            separator = "=" * 50
            
            entry = [
                f"\n{separator}",
                f"TIMESTAMP: {timestamp}",
                f"PROMPT: {prompt.strip()}",
                f"{separator}"
            ]
            
            if metrics:
                entry.append("METRICS SUMMARY:")
                entry.append(f"- Length: {metrics.get('character_count', 0)} chars")
                entry.append(f"- Time: {metrics.get('generation_time', 0.0):.2f}s")
                entry.append(f"- Coherence: {metrics.get('coherence_score', 0.0):.2f}")
                entry.append(f"- Sentiment: {metrics.get('sentiment', 0.0):.2f}")
                entry.append("-" * 20)
                
            entry.append("GENERATED OUTPUT:")
            entry.append(output.strip())
            entry.append(f"{separator}\n")
            
            with open(self.content_file, 'a', encoding='utf-8') as f:
                f.write("\n".join(entry))
                
        except Exception as e:
            print(f"Error logging content: {e}")
