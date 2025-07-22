import logging
import sys
from pathlib import Path
import io
import os

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration for the application"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # For Windows, set console to UTF-8 mode if possible
    if sys.platform.startswith('win'):
        try:
            # Try to set console to UTF-8 mode
            os.system('chcp 65001 > nul')
        except:
            pass
    
    # Create a custom formatter that handles Unicode gracefully
    class SafeFormatter(logging.Formatter):
        def format(self, record):
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # If Unicode fails, replace problematic characters
                msg = record.getMessage()
                if isinstance(msg, str):
                    record.msg = msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                return super().format(record)
    
    # Setup logging with safe formatter
    handler_file = logging.FileHandler(log_dir / "rag_system.log", encoding='utf-8')
    handler_file.setFormatter(SafeFormatter(log_format))
    
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(SafeFormatter(log_format))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=[handler_file, handler_console]
    )
    
    # Set specific logger levels
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)
