import argparse
import asyncio
import csv
import os
import sys
import urllib.request
from typing import List

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.training_pipeline import run_full_training_pipeline
from app.config import settings
from app.logging_config import logger

def fetch_csv(path_or_url: str) -> List[str]:
    """Reads a CSV file or URL and extracts all text into a list."""
    texts = []
    
    try:
        if path_or_url.startswith('http://') or path_or_url.startswith('https://'):
            print(f"Downloading CSV from {path_or_url}...")
            response = urllib.request.urlopen(path_or_url)
            lines = [l.decode('utf-8') for l in response.readlines()]
            reader = csv.reader(lines)
        else:
            print(f"Reading local CSV from {path_or_url}...")
            f = open(path_or_url, 'r', encoding='utf-8')
            reader = csv.reader(f)

        # Attempt to auto-detect the column containing the text. 
        # For simplicity, we assume the longest string in each row is the text.
        for row in reader:
            if not row:
                continue
            # Find the longest column string (usually the journal entry / tweet)
            longest_col = max(row, key=len)
            if len(longest_col) > 10: # Filter out short boolean/ID columns
                texts.append(longest_col.strip())
                
        if 'f' in locals():
            f.close()
            
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)
        
    return texts

async def main():
    parser = argparse.ArgumentParser(description="Train MindMesh AI unsupervised models on a real CSV dataset.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="https://raw.githubusercontent.com/dair-ai/emotion_dataset/master/data/sample.csv",
        help="Path or URL to a CSV file containing real journal entries or text data."
    )
    args = parser.parse_args()

    print(f"\n--- MindMesh AI Real Dataset Trainer ---")
    
    corpus_texts = fetch_csv(args.dataset)
    print(f"Successfully loaded {len(corpus_texts)} text documents from the dataset.")
    
    if len(corpus_texts) < 15:
        print("Dataset is too small! Please provide a dataset with at least 15 text entries.")
        sys.exit(1)
        
    print("\nTraining models (Text Embeddings, Emotion Clustering, Topic Discovery) on real text...")
    
    # Run the unsupervised pipeline!
    # Note: We pass the real 'corpus_texts' for the NLP models. 
    # The anomaly & clustering models will still use synthetic feature vectors unless we also pass 'features='.
    await run_full_training_pipeline(corpus=corpus_texts)
    
    print("\nTraining complete! Your real-data models have been saved to disk at:")
    print(settings.MODEL_SAVE_DIR)

if __name__ == "__main__":
    asyncio.run(main())
