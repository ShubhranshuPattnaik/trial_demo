# Setup logging
from config.logging_config import quick_setup
logger = quick_setup('test_log')

from embedders.PubMedBERT import PubMedBERTEmbedder
from utils.util import load_trials_data
from db.elastic_search import ElasticsearchVectorStore, process_and_index_trials


def test_complete_pipeline():
    """Test the complete pipeline from data loading to Elasticsearch indexing"""
    logger.info("=== Testing Complete Pipeline ===")
    
    try:
        # Load data
        logger.info("Loading trial data...")
        df = load_trials_data("./data/ctg-studies.csv")
        logger.info(f"Loaded {len(df)} trial records")
        
        if len(df) == 0:
            logger.error("No trial data found")
            return False
        
        # Initialize PubMedBERT embedders
        logger.info("Initializing PubMedBERT embedders...")
        embedder = PubMedBERTEmbedder()
        logger.info("PubMedBERT embedders initialized")
        
        # Process trials (use small subset for testing)
        logger.info("Processing trials and generating embeddings...")
        test_df = df.head(3)  # Use only first 3 trials for testing
        results = embedder.process_trials_batch(test_df, pooling_method='mean')
        logger.info(f"Generated embeddings for {len(results)} trials")
        
        # Initialize Elasticsearch
        logger.info("Initializing Elasticsearch...")
        es_store = ElasticsearchVectorStore(index_name="integration_test")
        logger.info("Elasticsearch client initialized")
        
        # Create index
        logger.info("Creating Elasticsearch index...")
        success = es_store.create_index_with_mapping(embedding_dims=768)
        if success:
            logger.info("Index created successfully")
        else:
            logger.error("Failed to create index")
            return False
        
        # Index trials
        logger.info("Indexing trials in Elasticsearch...")
        indexing_stats = process_and_index_trials(results, es_store, test_df)
        logger.info("Indexing completed:")
        logger.info(f"  - Successfully indexed: {indexing_stats['success_count']}")
        logger.info(f"  - Failed: {indexing_stats['failure_count']}")
        
        if indexing_stats['success_count'] == 0:
            logger.error("No trials were indexed successfully")
            return False
        
        # Test vector similarity search
        logger.info("Testing vector similarity search...")
        query_vector = results[0]['embedding']  # Use first trial's embedding as query
        similar_trials = es_store.search_similar_trials(
            query_vector=query_vector,
            top_k=3
        )
        
        logger.info(f"Vector search successful, found {len(similar_trials)} results:")
        for i, trial in enumerate(similar_trials, 1):
            logger.debug(f"  {i}. NCT: {trial['nct_number']}, Score: {trial['score']:.4f}")
        
        # Test text search
        logger.info("Testing text-based search...")
        text_results = es_store.search_by_text(
            query_text="stroke",
            top_k=3
        )
        
        logger.info(f"Text search successful, found {len(text_results)} results:")
        for i, trial in enumerate(text_results, 1):
            logger.debug(f"  {i}. NCT: {trial['nct_number']}, Score: {trial['score']:.4f}")
        
        # Test cross-modal search (text query -> vector search)
        logger.info("Testing cross-modal search...")
        # Generate embedding for a text query
        query_text = "myocardial infarction rehabilitation"
        query_embedding = embedder.generate_embedding(query_text)
        
        # Search using the generated embedding
        cross_modal_results = es_store.search_similar_trials(
            query_vector=query_embedding,
            top_k=3
        )
        
        logger.info(f"Cross-modal search successful, found {len(cross_modal_results)} results:")
        for i, trial in enumerate(cross_modal_results, 1):
            logger.debug(f"  {i}. NCT: {trial['nct_number']}, Score: {trial['score']:.4f}")
        
        logger.info("=== Complete pipeline test passed! ===")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return False
    finally:
        # Clean up: delete test index
        try:
            logger.info("Cleaning up integration test index...")
            es_store = ElasticsearchVectorStore(index_name="integration_test")
            if es_store.delete_index():
                logger.info("Integration test index deleted successfully")
            else:
                logger.warning("Failed to delete integration test index")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")

def test_performance():
    """Test performance with different batch sizes"""
    logger.info("=== Testing Performance ===")
    
    try:
        import time
        
        # Load data
        df = load_trials_data("./data/ctg-studies.csv")
        embedder = PubMedBERTEmbedder()
        
        # Test different batch sizes
        batch_sizes = [1, 2, 5]
        
        for batch_size in batch_sizes:
            logger.info(f"Testing with batch size: {batch_size}")
            
            start_time = time.time()
            
            # Process batch
            test_df = df.head(batch_size)
            results = embedder.process_trials_batch(test_df, pooling_method='mean')
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            logger.info(f"  - Processed {len(results)} trials in {processing_time:.2f} seconds")
            logger.debug(f"  - Average time per trial: {processing_time/len(results):.2f} seconds")
        
        logger.info("Performance test completed")
        return True
        
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in various scenarios"""
    logger.info("=== Testing Error Handling ===")
    
    try:
        embedder = PubMedBERTEmbedder()
        
        # Test 1: Empty text
        logger.info("Testing empty text handling...")
        try:
            embedding = embedder.generate_embedding("")
            logger.info("Empty text handled gracefully")
        except Exception as e:
            logger.warning(f"Empty text caused error: {e}")
        
        # Test 2: Very long text
        logger.info("Testing long text handling...")
        long_text = "This is a very long clinical trial description. " * 100
        try:
            embedding = embedder.generate_embedding(long_text)
            logger.info(f"Long text handled gracefully, embedding dimension: {len(embedding)}")
        except Exception as e:
            logger.warning(f"Long text caused error: {e}")
        
        # Test 3: Special characters
        logger.info("Testing special characters...")
        special_text = "Clinical trial with special chars: @#$%^&*()_+-=[]{}|;':\",./<>?"
        try:
            embedding = embedder.generate_embedding(special_text)
            logger.info(f"Special characters handled gracefully, embedding dimension: {len(embedding)}")
        except Exception as e:
            logger.warning(f"Special characters caused error: {e}")
        
        logger.info("Error handling tests completed")
        return True
        
    except Exception as e:
        logger.error(f"Error handling test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_complete_pipeline()
    success2 = test_performance()
    success3 = test_error_handling()
    
    if success1 and success2 and success3:
        logger.info("All integration tests passed successfully!")
    else:
        logger.error("Some integration tests failed!")
