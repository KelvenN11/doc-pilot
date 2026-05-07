from pathlib import Path

# Project root = parent folder of /src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "course_notes"

# Local embedding model.
# Good default for small local RAG.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Number of chunks to retrieve for each query.
DEFAULT_TOP_K = 5