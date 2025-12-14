import sys
import os
sys.path.append('src')

from data_preprocessing import LoanDataPreprocessor
from model_training import run_complete_training
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main pipeline execution"""
    logger.info("Starting Bank Loan Default Prediction Pipeline...")
    
    try:
        # Step 1: Run EDA (optional - run the notebook separately)
        logger.info("Step 1: Run EDA notebook separately to understand the data")
        
        # Step 2: Run complete training pipeline
        logger.info("Step 2: Starting model training...")
        run_complete_training()
        
        logger.info("Pipeline completed successfully!")
        logger.info("Next steps:")
        logger.info("1. Check models/ directory for saved models and results")
        logger.info("2. Review SHAP plots for model interpretability")
        logger.info("3. Proceed to deployment with FastAPI and Streamlit")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()