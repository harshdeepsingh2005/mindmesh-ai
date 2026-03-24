import asyncio
import os
import sys

# Ensure backend directory is in path
sys.path.insert(0, os.path.dirname(__file__))

from app.services.training_pipeline import run_full_training_pipeline
from app.config import settings

async def main():
    print(f"Model directory will be: {settings.MODEL_SAVE_DIR}")
    print("Running training pipeline to generate synthetic data and save models...")
    await run_full_training_pipeline()
    
    print("\nChecking generated files in saved_models directory:")
    if os.path.exists(settings.MODEL_SAVE_DIR):
        print(os.listdir(settings.MODEL_SAVE_DIR))
    else:
        print("Directory was not created!")

if __name__ == "__main__":
    asyncio.run(main())
