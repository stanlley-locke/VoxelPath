
import sys
print(f"Python: {sys.version}")
try:
    import numpy
    print("Numpy: OK")
except ImportError as e:
    print(f"Numpy: FAIL ({e})")

try:
    import sentence_transformers
    print("Sentence Transformers: OK")
except ImportError as e:
    print(f"Sentence Transformers: FAIL ({e})")
