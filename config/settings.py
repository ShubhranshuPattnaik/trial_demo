"""
Configuration management for TrialMatcher RAG project
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for managing all project settings"""
    
    # Elasticsearch Configuration
    ELASTICSEARCH_URL: str = os.getenv('ELASTICSEARCH_URL')
    ELASTICSEARCH_API_KEY: str = os.getenv('ELASTICSEARCH_API_KEY')
    ELASTICSEARCH_INDEX_NAME: str = os.getenv('ELASTICSEARCH_INDEX_NAME', 'trial_embeddings')

    # Field Mapping Configuration for Elasticsearch
    # TODO: Need modificaitons for flexible data structure
    FIELD_MAPPINGS: dict = {
        'id_field': os.getenv('ID_FIELD', 'NCT Number'),
        'title_field': os.getenv('TITLE_FIELD', 'Study Title'),
        'conditions_field': os.getenv('CONDITIONS_FIELD', 'Conditions'),
        'interventions_field': os.getenv('INTERVENTIONS_FIELD', 'Interventions'),
        'phases_field': os.getenv('PHASES_FIELD', 'Phases'),
        'study_type_field': os.getenv('STUDY_TYPE_FIELD', 'Study Type'),
        'text_fields': os.getenv('TEXT_FIELDS', 'Study Title,Conditions,Interventions').split(',')
    }

    # Embedder Type Selection
    # Options: 'pubmedbert', 'sentence-transformer'
    EMBEDDER_TYPE: str = os.getenv('EMBEDDER_TYPE', 'pubmedbert')
    
    # PubMedBERT Configuration
    PUBMEDBERT_MODEL_NAME: str = os.getenv('PUBMEDBERT_MODEL_NAME', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    EMBEDDING_DIMENSION: int = int(os.getenv('EMBEDDING_DIMENSION', '768'))
    POOLING_METHOD: str = os.getenv('POOLING_METHOD', 'mean')

    # Sentence Transformer Configuration
    SENTENCE_TRANSFORMER_MODEL_NAME: str = os.getenv('SENTENCE_TRANSFORMER_MODEL_NAME', 'neuml/pubmedbert-base-embeddings')
    SENTENCE_EMBEDDING_DIMENSION: int = int(os.getenv('SENTENCE_EMBEDDING_DIMENSION', '768'))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(levelname)s - %(message)s')
    LOG_TO_FILE: bool = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_TO_CONSOLE: bool = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
    LOG_DIRECTORY: str = os.getenv('LOG_DIRECTORY', 'logs')
    
    # Data Configuration
    DATA_FILE_PATH: str = os.getenv('DATA_FILE_PATH', 'data/ctg-studies.csv')
    EMBEDDINGS_FILE_PATH: str = os.getenv('EMBEDDINGS_FILE_PATH', 'data/trial_embeddings.npy')
    
    # Search Configuration
    DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', '10'))
    DEFAULT_MIN_SCORE: float = float(os.getenv('DEFAULT_MIN_SCORE', '0.0'))
    SEARCH_TIMEOUT: int = int(os.getenv('SEARCH_TIMEOUT', '30'))
    
    # Performance Configuration
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '32'))
    MAX_TEXT_LENGTH: int = int(os.getenv('MAX_TEXT_LENGTH', '512'))

    @classmethod
    def get_log_level(cls) -> int:
        """Get logging level as integer"""
        import logging
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_map.get(cls.LOG_LEVEL.upper(), logging.INFO)
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate Elasticsearch settings
        if not cls.ELASTICSEARCH_URL:
            errors.append("ELASTICSEARCH_URL is required")
        if not cls.ELASTICSEARCH_API_KEY:
            errors.append("ELASTICSEARCH_API_KEY is required")
        
        # Validate embedding dimension
        if cls.EMBEDDING_DIMENSION <= 0:
            errors.append("EMBEDDING_DIMENSION must be positive")
        
        # Validate embedder type
        if cls.EMBEDDER_TYPE not in ['pubmedbert', 'sentence-transformer']:
            errors.append("EMBEDDER_TYPE must be one of: pubmedbert, sentence-transformer")
        
        # Validate pooling method
        if cls.POOLING_METHOD not in ['mean', 'max', 'cls']:
            errors.append("POOLING_METHOD must be one of: mean, max, cls")
        
        # Validate log level
        if cls.LOG_LEVEL.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            errors.append("LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# Global configuration instance
config = Config()
