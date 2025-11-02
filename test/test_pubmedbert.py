# Setup unified logging
from config.logging_config import quick_setup
logger = quick_setup('test_log')

from embedders.PubMedBERT import PubMedBERTEmbedder
from utils.util import load_trials_data
import numpy as np

def test_pubmedbert_embedder():
    """Test PubMedBERT embedders functionality"""
    logger.info("=== Testing PubMedBERT Embedder ===")
    
    try:
        # Test embedders initialization
        logger.info("Testing embedders initialization...")
        embedder = PubMedBERTEmbedder()
        logger.info("PubMedBERT embedders initialized successfully")
        
        # Test data loading
        logger.info("Testing data loading...")
        df = load_trials_data("./data/ctg-studies.csv")
        logger.info(f"Loaded {len(df)} trial records")
        
        if len(df) == 0:
            logger.error("No trial data found")
            return False
        
        # Test text formatting
        logger.info("Testing text formatting...")
        sample_row = df.iloc[0]
        formatted_text = embedder.format_trial_text(sample_row)
        logger.info("Text formatting successful")
        logger.debug(f"  - Sample formatted text: {formatted_text[:100]}...")
        
        # Test single embedding generation
        logger.info("Testing single embedding generation...")
        test_text = "This is a test clinical trial for stroke rehabilitation"
        embedding = embedder.generate_embedding(test_text)
        logger.info("Single embedding generated successfully")
        logger.debug(f"  - Embedding dimension: {len(embedding)}")
        logger.debug(f"  - Embedding type: {type(embedding)}")
        logger.debug(f"  - Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        
        # Test single trial processing
        logger.info("Testing single trial processing...")
        trial_result = embedder.process_single_trial(sample_row)
        logger.info("Single trial processed successfully")
        logger.debug(f"  - NCT Number: {trial_result['nct_number']}")
        logger.debug(f"  - Text length: {len(trial_result['text'])}")
        logger.debug(f"  - Embedding dimension: {trial_result['embedding_dim']}")
        
        # Test batch processing (small batch)
        logger.info("Testing batch processing...")
        small_df = df.head(2)  # Process only first 2 trials for testing
        batch_results = embedder.process_trials_batch(small_df)
        logger.info("Batch processing successful")
        logger.debug(f"  - Processed {len(batch_results)} trials")
        logger.debug(f"  - All embeddings have dimension: {batch_results[0]['embedding_dim']}")
        
        # Test embedding matrix generation
        logger.info("Testing embedding matrix generation...")
        embedding_matrix = embedder.get_embedding_matrix(batch_results)
        logger.info("Embedding matrix generated successfully")
        logger.debug(f"  - Matrix shape: {embedding_matrix.shape}")
        logger.debug(f"  - Matrix type: {type(embedding_matrix)}")
        
        # Test different pooling methods
        logger.info("Testing different pooling methods...")
        pooling_methods = ['mean', 'max', 'cls']
        for method in pooling_methods:
            embedding = embedder.generate_embedding(test_text, pooling_method=method)
            logger.debug(f"  - {method} pooling: dimension {len(embedding)}, range [{embedding.min():.4f}, {embedding.max():.4f}]")
        logger.info("All pooling methods tested successfully")
        
        logger.info("=== All PubMedBERT tests passed! ===")
        return True
        
    except Exception as e:
        logger.error(f"PubMedBERT test failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return False

def test_embedding_quality():
    """Test embedding quality and consistency"""
    logger.info("=== Testing Embedding Quality ===")
    
    try:
        embedder = PubMedBERTEmbedder()
        
        # Test 1: Same text should produce same embedding
        logger.info(" Testing embedding consistency...")
        text1 = "Stroke rehabilitation clinical trial"
        embedding1 = embedder.generate_embedding(text1)
        embedding2 = embedder.generate_embedding(text1)
        
        # Check if embeddings are identical (they should be)
        if np.allclose(embedding1, embedding2):
            logger.info("Embeddings are consistent for identical text")
        else:
            logger.error("Embeddings are not consistent for identical text")
            return False
        
        # Test 2: Different texts should produce different embeddings
        logger.info(" Testing embedding differentiation...")
        text2 = "Cancer treatment clinical trial"
        embedding3 = embedder.generate_embedding(text2)
        
        # Check if embeddings are different
        similarity = np.dot(embedding1, embedding3) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding3))
        logger.debug(f"  - Cosine similarity between different texts: {similarity:.4f}")
        
        if similarity < 0.95:  # Should be reasonably different
            logger.info("Different texts produce different embeddings")
        else:
            logger.warning("Different texts produce too similar embeddings")
        
        # Test 3: Similar texts should have higher similarity
        logger.info(" Testing semantic similarity...")
        text3 = "Stroke recovery clinical study"
        embedding4 = embedder.generate_embedding(text3)
        
        similarity_stroke = np.dot(embedding1, embedding4) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding4))
        logger.debug(f"  - Similarity between stroke-related texts: {similarity_stroke:.4f}")
        logger.debug(f"  - Similarity between stroke and cancer: {similarity:.4f}")
        
        if similarity_stroke > similarity:
            logger.info("Semantically similar texts have higher similarity")
        else:
            logger.warning("Semantic similarity not properly captured")
        
        logger.info("=== Embedding quality tests completed ===")
        return True
        
    except Exception as e:
        logger.error(f"Embedding quality test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_pubmedbert_embedder()
    success2 = test_embedding_quality()
    
    if success1 and success2:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed!")
