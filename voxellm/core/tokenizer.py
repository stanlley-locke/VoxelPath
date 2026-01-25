import re
import math
from typing import List, Dict
from collections import Counter, defaultdict
from .structures import TextAnalysisResult

class IntelligentTokenizer:
    """Advanced tokenizer with punctuation handling and ASCII sanitization"""
    # Regex patterns for sophisticated tokenization
    PUNCTUATION_PATTERN = r'([\.,!?;:"\'\-–—()\[\]{}])'
    WORD_PATTERN = r'(\b[A-Za-z]+\b)'
    NUMBER_PATTERN = r'(\b\d+\b)'
    
    # ASCII safe mapping for non-standard characters
    ASCII_SAFE_MAP = {
        '—': '-',  # Em dash to hyphen
        '–': '-',  # En dash to hyphen
        '“': '"',  # Smart quotes
        '”': '"',
        '‘': "'",
        '’': "'",
        '…': '...',  # Ellipsis
        '«': '"',
        '»': '"',
        'é': 'e',
        'è': 'e',
        'ê': 'e',
        'ë': 'e',
        'á': 'a',
        'à': 'a',
        'â': 'a',
        'ä': 'a',
        'í': 'i',
        'ì': 'i',
        'î': 'i',
        'ï': 'i',
        'ó': 'o',
        'ò': 'o',
        'ô': 'o',
        'ö': 'o',
        'ú': 'u',
        'ù': 'u',
        'û': 'u',
        'ü': 'u',
        'ñ': 'n',
        'ç': 'c'
    }
    
    def __init__(self):
        # Combined pattern that captures words, numbers, punctuation, and newlines
        self.token_pattern = re.compile(
            f'{self.WORD_PATTERN}|{self.NUMBER_PATTERN}|{self.PUNCTUATION_PATTERN}|(\n)'
        )
        # Define punctuation sets for detokenization
        self._punctuation_no_space_before = set('.,!?;:"\'') | set(')-]}>')
        self._punctuation_no_space_after = set('([{<')
        
    def sanitize_text(self, text: str) -> str:
        """Convert non-standard characters to ASCII-safe equivalents"""
        for old, new in self.ASCII_SAFE_MAP.items():
            text = text.replace(old, new)
        # Remove any remaining non-ASCII characters, but keep newlines
        text = ''.join(char for char in text if 31 < ord(char) < 127 or ord(char) == 10)
        return text
        
    def tokenize(self, text: str) -> List[str]:
        """Intelligent tokenization preserving punctuation as separate tokens"""
        text = self.sanitize_text(text)
        tokens = []
        # Use regex to split while keeping delimiters
        matches = self.token_pattern.finditer(text)
        for match in matches:
            token = match.group()
            if token and token.strip():  # Filter out empty matches and whitespace
                tokens.append(token.lower())  # Case normalization
                
        # Post-processing: handle contractions
        processed_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] in ["'", "’"] and tokens[i+1] in ['t', 's', 'm', 'd', 'll', 've', 're']:
                # Merge contractions like "don't", "it's", etc.
                if processed_tokens:
                    processed_tokens[-1] = processed_tokens[-1] + "'" + tokens[i+1]
                i += 2
            else:
                processed_tokens.append(tokens[i])
                i += 1
                
        return processed_tokens
        
    def detokenize(self, tokens: List[str]) -> str:
        """Reconstruct text from tokens with proper spacing and capitalization"""
        result = []
        capitalize_next = True
        
        for i, token in enumerate(tokens):
            # Handle sentence boundaries
            if token in ['.', '!', '?'] and i < len(tokens) - 1:
                result.append(token)
                capitalize_next = True
            elif capitalize_next and token.isalpha():
                result.append(token.capitalize())
                capitalize_next = False
            elif token in self._punctuation_no_space_before:
                # Attach punctuation without space
                if result:
                    result[-1] = result[-1] + token
            elif result and result[-1] and result[-1][-1] in self._punctuation_no_space_after:
                # Attach to opening punctuation without space
                result[-1] = result[-1] + token
            else:
                # Add space before word/number
                if result:
                    result.append(' ' + token)
                else:
                    result.append(token)
                    capitalize_next = False
            
            # Handle special cases for apostrophes
            if token == "'" and i > 0 and i < len(tokens) - 1:
                if result:
                    result[-1] = result[-1] + "'"
                    continue
            
        # Capitalize first letter if needed
        if result and result[0]:
            result[0] = result[0].capitalize()
            
        return ''.join(result).strip()
        
    def analyze_text(self, tokens: List[str]) -> TextAnalysisResult:
        """Comprehensive text analysis with multiple quality metrics"""
        result = TextAnalysisResult()
        
        # Calculate average word length (excluding punctuation)
        words = [token for token in tokens if token.isalpha()]
        if words:
            word_lengths = [len(word) for word in words]
            result.avg_word_length = sum(word_lengths) / len(word_lengths)
            
        # Calculate repetition rate
        if tokens:
            unique_tokens = set(tokens)
            result.repetition_rate = 1.0 - (len(unique_tokens) / len(tokens))
            
        # Calculate sentence complexity
        sentence_delimiters = ['.', '!', '?']
        sentences = []
        current_sentence = []
        
        for token in tokens:
            current_sentence.append(token)
            if token in sentence_delimiters:
                sentences.append(current_sentence)
                current_sentence = []
                
        if current_sentence:  # Add last incomplete sentence
            sentences.append(current_sentence)
            
        if sentences:
            sentence_lengths = [len(s) for s in sentences]
            result.sentence_complexity = sum(sentence_lengths) / len(sentence_lengths)
            
        # Calculate vocabulary diversity (type-token ratio)
        if tokens:
            # Moving window diversity
            window_size = min(50, len(tokens))
            diversity_scores = []
            
            for i in range(0, len(tokens), window_size // 2):
                window = tokens[i:i+window_size]
                if window:
                    unique_in_window = len(set(window))
                    diversity_scores.append(unique_in_window / len(window))
                    
            if diversity_scores:
                result.diversity_score = sum(diversity_scores) / len(diversity_scores)
                
        # Calculate topic consistency (simplified version)
        # Group related words by common prefixes or semantic similarity
        if words:
            # Simple prefix-based grouping
            prefix_groups = defaultdict(list)
            for word in words:
                if len(word) >= 3:
                    prefix = word[:3]
                    prefix_groups[prefix].append(word)
                    
            # Calculate consistency score based on dominant groups
            if prefix_groups:
                group_sizes = [len(group) for group in prefix_groups.values()]
                max_group = max(group_sizes)
                result.topic_consistency = max_group / len(words)
                
        # Grammatical correctness estimation (very simplified)
        # Check for basic patterns like subject-verb agreement
        result.grammatical_correctness = self._estimate_grammatical_correctness(tokens)
        
        # Coherence score (transitions between sentences)
        result.coherence_score = self._calculate_coherence_score(sentences)
        
        # Perplexity estimation
        result.perplexity = self._estimate_perplexity(tokens)
        
        # New Metrics
        result.flesch_reading_ease = self._calculate_flesch_reading_ease(tokens, sentences)
        result.sentiment = self._calculate_sentiment(tokens)
        
        return result
        
    def _estimate_grammatical_correctness(self, tokens: List[str]) -> float:
        """Simple grammatical correctness estimation"""
        score = 1.0
        
        # Check for proper sentence structure
        sentence_starts = []
        sentence_ends = []
        
        for i, token in enumerate(tokens):
            if i == 0 or tokens[i-1] in ['.', '!', '?']:
                sentence_starts.append(token)
            if token in ['.', '!', '?']:
                sentence_ends.append(token)
                
        # Check ratio of sentence starts to ends
        if sentence_starts and sentence_ends:
            ratio = len(sentence_ends) / len(sentence_starts)
            score *= max(0.5, min(1.5, ratio))  # Normalize to 0.5-1.5 range
            
        # Check for common grammatical errors
        error_patterns = [
            (r'\bthe the\b', 0.1),
            (r'\ba a\b', 0.1),
            (r'\ban an\b', 0.1),
            (r'\bis is\b', 0.2),
            (r'\bare are\b', 0.2),
            (r'\bwas was\b', 0.2),
            (r'\bwere were\b', 0.2),
        ]
        
        text = self.detokenize(tokens).lower()
        for pattern, penalty in error_patterns:
            if re.search(pattern, text):
                score *= (1 - penalty)
                
        return max(0.0, min(1.0, score))
        
    def _calculate_coherence_score(self, sentences: List[List[str]]) -> float:
        """Calculate coherence between sentences"""
        if len(sentences) < 2:
            return 1.0
            
        coherence_scores = []
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i]
            next_sentence = sentences[i+1]
            
            # Extract nouns and verbs from each sentence
            current_words = [w for w in current_sentence if w.isalpha() and len(w) > 2]
            next_words = [w for w in next_sentence if w.isalpha() and len(w) > 2]
            
            # Simple overlap scoring
            common_words = set(current_words) & set(next_words)
            if current_words and next_words:
                overlap = len(common_words) / (len(current_words) + len(next_words))
                coherence_scores.append(overlap)
                
        if coherence_scores:
            return min(1.0, sum(coherence_scores) / len(coherence_scores) * 2)  # Scale up
        return 0.5
        
    def _estimate_perplexity(self, tokens: List[str]) -> float:
        """Simple perplexity estimation based on token frequency"""
        if not tokens:
            return float('inf')
            
        # Count token frequencies
        freq = Counter(tokens)
        total = len(tokens)
        
        # Calculate entropy
        entropy = 0.0
        for token, count in freq.items():
            p = count / total
            entropy -= p * math.log(p + 1e-10)  # Add small epsilon to avoid log(0)
            
        # Perplexity = 2^entropy
        return math.exp(entropy)

    def _calculate_flesch_reading_ease(self, tokens: List[str], sentences: List[List[str]]) -> float:
        """Calculate Flesch Reading Ease score"""
        if not tokens or not sentences:
            return 0.0
            
        total_words = len([t for t in tokens if t.isalpha()])
        total_sentences = len(sentences)
        
        if total_words == 0 or total_sentences == 0:
            return 0.0
            
        # Count syllables (heuristic)
        total_syllables = 0
        for token in tokens:
            if token.isalpha():
                total_syllables += self._count_syllables(token)
                
        # Formula: 206.835 - 1.015(total words / total sentences) - 84.6(total syllables / total words)
        score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
        return max(0.0, min(100.0, score))

    def _count_syllables(self, word: str) -> int:
        """Simple syllable counter heuristic"""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count
        
    def _calculate_sentiment(self, tokens: List[str]) -> float:
        """Simple heuristic sentiment analysis"""
        # Basic lexicon
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'beautiful',
            'happy', 'joy', 'love', 'best', 'fantastic', 'success', 'bright',
            'intelligent', 'smart', 'creative', 'innovative', 'efficient',
            'powerful', 'strong', 'peace', 'victory', 'friend', 'hope'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'ugly',
            'sad', 'hate', 'fail', 'failure', 'dark', 'pain', 'anger',
            'enemy', 'fear', 'destroy', 'broken', 'poor', 'stupid',
            'waste', 'problem', 'error', 'wrong', 'sick', 'weak'
        }
        
        score = 0
        count = 0
        
        for token in tokens:
            if token in positive_words:
                score += 1
                count += 1
            elif token in negative_words:
                score -= 1
                count += 1
        
        # Normalize to -1.0 to 1.0 range, but dampen extremes for short texts
        if count == 0:
            return 0.0
            
        normalized = score / max(1, count)
        return max(-1.0, min(1.0, normalized))
