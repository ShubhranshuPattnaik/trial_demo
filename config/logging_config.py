"""
Unified logging configuration for the TrialMatcher RAG project
"""

import logging
import os
from datetime import datetime
from typing import Optional
from config.settings import config

def setup_logging(
    level: int = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    add_timestamp: bool = True
) -> logging.Logger:
    """
    Setup unified logging configuration using environment variables
    
    Args:
        level: Logging level (default: from config)
        log_file: Log file path (default: auto-generated)
        format_string: Custom format string (default: from config)
        add_timestamp: Whether to add timestamp to log filename
        
    Returns:
        Configured logger instance
    """
    
    # Use config values if not provided
    if level is None:
        level = config.get_log_level()
    
    if format_string is None:
        format_string = config.LOG_FORMAT
    
    # Auto-generate log file name if not provided
    if log_file is None:
        # Get caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        caller_name = frame.f_globals.get('__name__', 'unknown')
        if caller_name == '__main__':
            # Use the script filename
            caller_name = os.path.basename(frame.f_code.co_filename).replace('.py', '')
        
        # Create logs directory if it doesn't exist
        logs_dir = config.LOG_DIRECTORY
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        
        # Generate log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if add_timestamp else ''
        log_file = os.path.join(logs_dir, f"{caller_name}{'_' + timestamp if timestamp else ''}.log")
    
    # Create handlers based on config
    handlers = []
    
    if config.LOG_TO_CONSOLE:
        handlers.append(logging.StreamHandler())
    
    if config.LOG_TO_FILE and log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override existing configuration
    )
    
    # Get logger for the calling module
    import inspect
    frame = inspect.currentframe().f_back
    caller_name = frame.f_globals.get('__name__', 'unknown')
    logger = logging.getLogger(caller_name)
    
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (default: current module)
        
    Returns:
        Logger instance
    """
    if name is None:
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'unknown')
    
    return logging.getLogger(name)

# Convenience function for quick setup
def quick_setup(script_name: str, level: int = None) -> logging.Logger:
    """
    Quick setup for scripts with auto-generated log file
    
    Args:
        script_name: Name of the script (used for log file naming)
        level: Logging level (default: from config)
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = config.LOG_DIRECTORY
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    log_file = os.path.join(logs_dir, f"{script_name}.log")
    
    return setup_logging(
        level=level,
        log_file=log_file,
        format_string=config.LOG_FORMAT
    )