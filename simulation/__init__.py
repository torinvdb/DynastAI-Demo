"""
DynastAI simulation package.
"""
# First, disable noisy loggers as early as possible
import logging
import os
import sys

# Set environment variables to make wandb quieter but still functional
os.environ['WANDB_SILENT'] = 'true'  # Silence wandb
os.environ['WANDB_DISABLE_SYMLINKS'] = 'true'  # Prevent wandb from creating symlinks
os.environ['WANDB_CONSOLE'] = 'off'  # Disable console output

# Configure noisy loggers to be less verbose but still functional
noisy_loggers = ['gql', 'gql.transport', 'gql.transport.requests']
for name in noisy_loggers:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    
    # Remove all handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    # Add a null handler to prevent propagation
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
            
    logger.addHandler(NullHandler())
