import numpy as np
from trial_rag import TrialRAG
from embedders.PubMedBERT import load_trials_data

from config.logging_config import quick_setup
logger = quick_setup('test_log')

def test_trail_rag():
    """Example usage of TrialRAG system"""
    import pandas as pd
    rag = TrialRAG()

    logger.info("=== Example 1: Adding data from CSV ===")
    df = load_trials_data()
    result = rag.add_data(df.head(3))
    logger.info(f"Example 1 result: {result}")

    logger.info("=== Example 2: Adding single trial ===")
    single_trial = {
        'NCT Number': 'NCT12345678',
        'Study Title': 'Example Clinical Trial',
        'Conditions': 'Example Condition',
        'Interventions': 'Example Intervention',
        'Phases': 'Phase II',
        'Study Type': 'Interventional'
    }
    result2 = rag.add_data(single_trial)
    logger.info(f"Example 2 result: {result2}")

    logger.info("=== Example 3: Adding multiple trials ===")
    multiple_trials = [
        {'NCT Number': 'NCT11111111', 'Study Title': 'Trial 1', 'Conditions': 'Condition 1',
         'Interventions': 'Intervention 1', 'Phases': 'Phase I', 'Study Type': 'Interventional'},
        {'NCT Number': 'NCT22222222', 'Study Title': 'Trial 2', 'Conditions': 'Condition 2',
         'Interventions': 'Intervention 2', 'Phases': 'Phase II', 'Study Type': 'Observational'}
    ]
    result3 = rag.add_data(multiple_trials)
    logger.info(f"Example 3 result: {result3}")

    logger.info("=== Search Examples ===")
    text_results = rag.search_by_text("stroke rehabilitation", top_k=3)
    logger.info(f"Text search results: {len(text_results)}")
    logger.info(f"{text_results}")
    vector_results = rag.search_by_text_with_vector("myocardial infarction", top_k=3)
    logger.info(f"Text-to-vector search results: {len(vector_results)}")


if __name__ == "__main__":
    test_trail_rag()