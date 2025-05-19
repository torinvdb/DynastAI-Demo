import os
import argparse
import logging
import logging.handlers
import signal
import sys
import json
import io
import warnings
import traceback
import builtins
import re
from pathlib import Path
from datetime import datetime
from functools import wraps

# Store original print function for selective filtering
original_print = builtins.print

# Create filter for unwanted output
UNWANTED_PATTERNS = [
    r'View Weave data at https://wandb\.ai',
    r'Logged in as Weights & Biases user',
    r'View Weave data',
    r'wandb:',
    r'Weights & Biases'
]
PATTERN_REGEX = re.compile('|'.join(UNWANTED_PATTERNS))

# Replace the built-in print function with a filtered version
def filtered_print(*args, **kwargs):
    if args:
        # Convert all arguments to strings and join them
        message = ' '.join(str(arg) for arg in args)
        # Check if the message contains any unwanted pattern
        if PATTERN_REGEX.search(message):
            return  # Skip printing
    
    # Otherwise, call the original print function
    original_print(*args, **kwargs)

# Set environment variables to make wandb quieter but still functional
os.environ['WANDB_SILENT'] = 'true'  # Silence wandb
os.environ['WANDB_DISABLE_SYMLINKS'] = 'true'  # Prevent wandb from creating symlinks
os.environ['WANDB_CONSOLE'] = 'off'  # Disable console output
os.environ['PYTHONWARNINGS'] = 'ignore'  # Silence warnings
os.environ['MPLBACKEND'] = 'Agg'  # Disable matplotlib font warnings

# Configure core logging immediately
logging.basicConfig(
    level=logging.WARNING,  # Default to WARNING level (quieter)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Immediately silence noisy loggers
for name in ['gql', 'gql.transport', 'gql.transport.requests']:
    logger = logging.getLogger(name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.handlers = []  # Remove all handlers
    logger.addHandler(logging.NullHandler())  # Add a null handler

# Now safely import other dependencies
import numpy as np
import pandas as pd

from .cards import load_cards
from .simulation import run_multi_strategy_simulations
from .analysis import analyze_simulations, compare_strategies, get_card_impact_analysis
from .visualization import create_visualizations
from .strategies import get_strategy_names
from .export import export_results_to_csv

# Get our module's logger
logger = logging.getLogger(__name__)

# Flag to track shutdown
_shutdown_requested = False

# ---- Logging utilities ----

class NullHandler(logging.Handler):
    """A logging handler that discards all log records."""
    def emit(self, record):
        pass

class SilenceLoggerFilter(logging.Filter):
    """A filter that silences specific log messages containing certain patterns."""
    def __init__(self, patterns=None):
        super().__init__()
        self.patterns = patterns or []
    
    def filter(self, record):
        # Check if message contains any of the patterns to silence
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern in self.patterns:
                if pattern in record.msg:
                    return False
        return True

class OutputFilter:
    """A simple context manager to filter stdout during execution of a block of code."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.null_file = None
        self.original_stdout = None
        
    def __enter__(self):
        if not self.verbose:
            self.null_file = open(os.devnull, 'w')
            self.original_stdout = sys.stdout
            sys.stdout = self.null_file
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout = self.original_stdout
            self.null_file.close()

# ---- Logging utilities ----

class NullHandler(logging.Handler):
    """A logging handler that discards all log records."""
    def emit(self, record):
        pass

class SilenceLoggerFilter(logging.Filter):
    """A filter that silences specific log messages containing certain patterns."""
    def __init__(self, patterns=None):
        super().__init__()
        self.patterns = patterns or []
    
    def filter(self, record):
        # Check if message contains any of the patterns to silence
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern in self.patterns:
                if pattern in record.msg:
                    return False
        return True

class StreamRedirector:
    """Context manager that redirects stdout/stderr to discard unwanted output."""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.original_stdout = None
        self.original_stderr = None
        self.null_out = None
        
    def __enter__(self):
        if not self.verbose:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            self.null_out = open(os.devnull, 'w')
            sys.stdout = self.null_out
            sys.stderr = self.null_out
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.null_out.close()

def quiet_unless_verbose(func):
    """
    Decorator to silence function output unless in verbose mode.
    Requires a 'verbose' parameter to be passed to the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        verbose = kwargs.get('verbose', False)
        with OutputFilter(verbose=verbose):
            return func(*args, **kwargs)
    return wrapper

def configure_logging(verbose=False, debug=False):
    """
    Configure logging for the application.
    
    Parameters
    ----------
    verbose : bool
        Whether to enable verbose output.
    debug : bool
        Whether to enable debug level logging.
    """
    # Set appropriate log level based on verbosity
    if debug:
        root_level = logging.DEBUG
    elif verbose:
        root_level = logging.INFO
    else:
        root_level = logging.WARNING
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Add filter to silence specific messages
    handler.addFilter(SilenceLoggerFilter([
        "Using gql", "GQL errors:", "transport.py", "Couldn't find user config"
    ]))
    
    root_logger.addHandler(handler)
    
    # Silence problematic libraries
    silent_loggers = [
        'gql', 'gql.transport', 'gql.transport.requests',
        'urllib3', 'requests', 'matplotlib', 'PIL', 'fontTools', 'numexpr'
    ]
    
    for name in silent_loggers:
        logger = logging.getLogger(name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.handlers = []  # Remove all handlers
        logger.addHandler(NullHandler())  # Add a null handler
    
    # In debug mode, enable logging for certain libraries
    if debug:
        # Enable weave and wandb logging at INFO level for debugging
        for name in ['weave', 'wandb']:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            logger.propagate = True
            # Clear existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
    
    # Set LLM strategy logger to appropriate level
    llm_logger = logging.getLogger('simulation.strategies.llm_strategies')
    if debug:
        llm_logger.setLevel(logging.DEBUG)
    elif verbose:
        llm_logger.setLevel(logging.INFO)
    else:
        llm_logger.setLevel(logging.WARNING)
        
    # Configure pandas and numpy for quiet output in non-verbose mode
    if not verbose:
        import pandas as pd
        import numpy as np
        pd.set_option('display.max_rows', 0)
        pd.set_option('display.max_columns', 0)
        pd.set_option('display.width', 0)
        np.set_printoptions(threshold=0, suppress=True)
        warnings.filterwarnings('ignore')
    
    root_logger.addHandler(handler)
    
    # Set appropriate level
    if verbose:
        root_logger.setLevel(logging.DEBUG)
    else:
        # In non-verbose mode, set root logger to WARNING to reduce output
        root_logger.setLevel(logging.WARNING)
    
    # Configure specifically noisy loggers
    noisy_loggers = [
        'gql', 'gql.transport', 'gql.transport.requests', 
        'wandb', 'weave', 'urllib3', 'requests', 'matplotlib', 
        'PIL', 'fontTools', 'numexpr'
    ]
    
    # Completely disable these loggers
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 100)  # Ensure nothing gets through
        logger.propagate = False  # Prevent propagation to parent loggers
        
        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Add null handler to prevent warnings
        logger.addHandler(NullHandler())
    
    # Configure project-specific loggers
    project_loggers = [
        'simulation.engine', 'simulation.cards', 'simulation.simulation'
    ]
    
    if not verbose:
        for logger_name in project_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)
        
        # Configure pandas and numpy
        pd.set_option('display.max_rows', 0)
        pd.set_option('display.max_columns', 0)
        pd.set_option('display.width', 0)
        np.set_printoptions(threshold=0, suppress=True)
        
        # Silence warnings
        warnings.filterwarnings('ignore')

def signal_handler(sig, frame):
    """
    Signal handler for graceful shutdown
    """
    global _shutdown_requested
    if not _shutdown_requested:
        logger.info("\nShutdown requested. Finishing current operations and saving partial results...")
        logger.info("Press Ctrl+C again to force exit immediately.")
        _shutdown_requested = True
        
        # Change handler for second press
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))

def save_partial_results(analyses, cards_df, output_dir):
    """
    Save partial analysis results as JSON
    
    Parameters:
    -----------
    analyses : dict
        Dictionary of analysis results
    cards_df : pandas DataFrame
        DataFrame containing card data
    output_dir : str
        Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save timestamp for partial results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Save analyses as JSON
        analyses_json = {}
        for strategy, analysis in analyses.items():
            if analysis:
                # Convert to serializable format
                analyses_json[strategy] = {
                    k: v for k, v in analysis.items() 
                    if k not in ['history'] and isinstance(v, (dict, list, str, int, float, bool, type(None)))
                }
        
        # Save the JSON
        partial_file = os.path.join(output_dir, f"partial_results_{timestamp}.json")
        with open(partial_file, 'w') as f:
            json.dump(analyses_json, f, indent=2)
        
        logger.info(f"Saved partial results to {partial_file}")
        
    except Exception as e:
        logger.error(f"Error saving partial results: {e}")

def main():
    """
    Main function to run the DynastAI simulation and analysis
    """
    global _shutdown_requested
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="DynastAI Game Analysis CLI")
    
    parser.add_argument('--cards_file', type=str, default=None, 
                        help='Path to the CSV file containing card data')
    parser.add_argument('--num_sims', type=int, default=500, 
                        help='Number of simulations per strategy (default: 500)')
    parser.add_argument('--max_turns', type=int, default=200, 
                        help='Maximum turns per game (default: 200)')
    parser.add_argument('--strategies', nargs='+', default=None, 
                        help='List of strategies to analyze (default: all)')
    parser.add_argument('--openrouter_system_prompt', type=str, default=None, 
                        help='System prompt for OpenRouter LLM')
    parser.add_argument('--openrouter_model', type=str, default=None, 
                        help='OpenRouter model name (default: google/gemini-2.5-flash-preview)')
    parser.add_argument('--openrouter_api_key', type=str, default=None, 
                        help='OpenRouter API key (if not set in environment variables)')
    parser.add_argument('--strategy_workers', type=int, default=None, 
                        help='Number of parallel workers for strategies (default: auto)')
    parser.add_argument('--sim_workers', type=int, default=10, 
                        help='Number of parallel workers for simulations per strategy (default: 10)')
    parser.add_argument('--output_dir', type=str, default='analysis_results', 
                        help='Output directory for results (default: analysis_results)')
    parser.add_argument('--skip_openrouter', action='store_true', 
                        help='Skip OpenRouter/LLM strategy')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable detailed output and statistics (default: quiet mode with minimal output)')
    parser.add_argument('--checkpoint_interval', type=int, default=0,
                        help='Save partial results every N simulations (0 to disable)')
    parser.add_argument('--sequential', action='store_true',
                        help='Run strategies sequentially instead of in parallel for better stability')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode for more detailed output and diagnostics')
    parser.add_argument('--api_request_delay', type=float, default=0.5,
                        help='Delay in seconds between API requests to avoid rate limiting (default: 0.5)')
    parser.add_argument('--max_api_retries', type=int, default=2,
                        help='Maximum number of retries for API requests (default: 2)')

    args = parser.parse_args()
    
    # Configure logging based on verbosity and debug mode
    configure_logging(verbose=args.verbose, debug=args.debug)
    
    # Print appropriate welcome message
    if args.verbose:
        print("\nDynastAI Game Analysis")
        print("----------------------")
        print("Press Ctrl+C once to gracefully abort and save partial results")
    else:
        print("DynastAI Game Analysis - Running simulations (press Ctrl+C to abort)")
    
    # Install signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Load cards data
        with OutputFilter(verbose=args.verbose):
            if args.cards_file:
                cards_df = load_cards(args.cards_file)
            else:
                # Look for the card file in the expected location
                card_file_path = 'cards/cards.csv'
                if os.path.exists(card_file_path):
                    logger.info(f"Using card file at {card_file_path}")
                    cards_df = load_cards(card_file_path)
                else:
                    logger.warning(f"Card file not found at {card_file_path}, using sample data")
                    cards_df = load_cards("sample_cards.csv")  # This will generate sample data
        
        # Print card data summary
        if args.verbose:
            print(f"\nLoaded {len(cards_df)} cards")
            
            # Count cards with dependencies
            if 'Requires_Card' in cards_df.columns:
                dependency_cards = cards_df[cards_df['Requires_Card'].notna()].shape[0]
                print(f"Found {dependency_cards} cards with dependencies on previous choices")
            
            if 'Requires_Metric' in cards_df.columns:
                metric_cards = cards_df[cards_df['Requires_Metric'].notna()].shape[0]
                print(f"Found {metric_cards} cards triggered by metric conditions")
        else:
            logger.info(f"Loaded {len(cards_df)} cards")
        
        # Determine strategies to analyze
        available_strategies = get_strategy_names()
        
        if args.skip_openrouter and 'openrouter' in available_strategies:
            available_strategies.remove('openrouter')
        
        if args.strategies:
            strategies = [s for s in args.strategies if s in available_strategies]
            if len(strategies) != len(args.strategies):
                invalid = set(args.strategies) - set(strategies)
                logger.warning(f"Ignoring invalid strategies: {invalid}")
                logger.info(f"Available strategies: {available_strategies}")
        else:
            strategies = available_strategies
        
        if args.verbose:
            print(f"\nAnalyzing strategies: {', '.join(strategies)}")
        else:
            logger.info(f"Analyzing strategies: {', '.join(strategies)}")
        
        # Prepare strategy-specific parameters
        strategy_params = {}
        for strategy in strategies:
            params = {}
            
            if strategy == 'openrouter':
                if args.openrouter_system_prompt:
                    params['system_prompt'] = args.openrouter_system_prompt
                if args.openrouter_model:
                    params['model'] = args.openrouter_model
                if args.openrouter_api_key:
                    params['api_key'] = args.openrouter_api_key
                # Add debug mode parameter
                if args.debug:
                    params['debug_mode'] = True
                # Add rate limiting parameters
                params['api_request_delay'] = args.api_request_delay
                params['max_api_retries'] = args.max_api_retries
            
            strategy_params[strategy] = params
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run simulations
        if args.verbose:
            print(f"\nRunning {args.num_sims} simulations per strategy...")
        else:
            logger.info(f"Running {args.num_sims} simulations per strategy...")
        
        # Force sequential mode if requested (overrides strategy_workers)
        if args.sequential:
            logger.info("Using sequential strategy execution mode for better stability")
            args.strategy_workers = 1
        
        all_results = run_multi_strategy_simulations(
            cards_df,
            strategy_params,
            num_sims=args.num_sims,
            max_turns=args.max_turns,
            strategy_workers=args.strategy_workers,
            sim_workers=args.sim_workers,
            verbose=args.verbose
        )
        
        # Analyze results
        analyses = {}
        if not _shutdown_requested:
            # Print minimal output even in non-verbose mode to show progress
            print("\nAnalyzing simulation results...")
            
            for strategy, results in all_results.items():
                if not results:
                    logger.warning(f"No results for {strategy} to analyze")
                    continue
                    
                # Use OutputFilter to completely suppress output for non-verbose mode
                with OutputFilter(verbose=args.verbose):
                    # Analyze the results
                    analysis = analyze_simulations(results, verbose=args.verbose)
                    analyses[strategy] = analysis
                    
                    # Print basic statistics only in verbose mode
                    if analysis and args.verbose:
                        stats = analysis['basic_stats']
                        print(f"\n{strategy.replace('_', ' ').title()} Strategy Results:")
                        print(f"  Mean turns survived: {stats['mean_turns']:.2f}")
                        print(f"  Median turns survived: {stats['median_turns']:.2f}")
                        print(f"  Standard deviation: {stats['std_dev']:.2f}")
                        print(f"  Min turns: {stats['min_turns']}")
                        print(f"  Max turns: {stats['max_turns']}")
                    
                    # Print most common end reasons
                    if 'end_reasons' in analysis:
                        print("\n  Most common end reasons:")
                        for reason, count in sorted(analysis['end_reasons'].items(), 
                                                key=lambda x: x[1], reverse=True)[:3]:
                            if reason:
                                total = sum(analysis['end_reasons'].values())
                                print(f"    {reason}: {count} games ({count/total*100:.1f}%)")
                        
                        # Print top risky cards
                        if 'risky_cards' in analysis:
                            print("\n  Top 5 riskiest cards:")
                            top_cards = sorted(analysis['risky_cards'].items(), 
                                            key=lambda x: x[1], reverse=True)[:5]
                            for card_key, count in top_cards:
                                print(f"    Card {card_key}: {count} games")
                    elif analysis:
                        # Just log basic info
                        logger.info(f"{strategy.replace('_', ' ').title()} - Mean turns: {analysis['basic_stats']['mean_turns']:.2f}, Median: {analysis['basic_stats']['median_turns']:.2f}")
            
            # Compare strategies
            if len(analyses) > 1:
                with OutputFilter(verbose=args.verbose):
                    comparison = compare_strategies(analyses, verbose=args.verbose)
                    
                    # Only print detailed comparison in verbose mode
                    if comparison and args.verbose:
                        print("\nStrategy Comparison:")
                        
                        if 'rankings' in comparison:
                            print(f"\n  Best strategies by mean turns survived:")
                            for i, strategy in enumerate(comparison['rankings']['by_mean_turns'][:3], 1):
                                mean_turns = comparison['mean_turns'].get(strategy, 0)
                                print(f"    {i}. {strategy.replace('_', ' ').title()}: {mean_turns:.2f} turns")
                            
                            print(f"\n  Best strategies by survival rate at turn 50:")
                            for i, strategy in enumerate(comparison['rankings']['by_survival_rate'][:3], 1):
                                survival_rate = comparison['survival_rates'].get(strategy, {}).get('turn_50', 0)
                                print(f"    {i}. {strategy.replace('_', ' ').title()}: {survival_rate*100:.1f}%")
                    elif comparison:
                        # Log basic comparison info
                        logger.info(f"Best strategy by mean turns: {comparison['rankings']['by_mean_turns'][0]}")
            
            # Analyze card impact
            with OutputFilter(verbose=args.verbose):
                card_impact = get_card_impact_analysis(cards_df, verbose=args.verbose)
                
                # Print card impact analysis only in verbose mode
                if not card_impact.empty and args.verbose:
                    print("\nCard Impact Analysis:")
                    print("  Top 5 highest impact cards:")
                    for i, (_, card) in enumerate(card_impact.head(5).iterrows(), 1):
                        print(f"    {i}. Card {card['Card_ID']} ({card['Character']}): "
                              f"Impact magnitude {card['Max_Total_Impact']:.1f}")
            
            # Create visualizations and export results
            with OutputFilter(verbose=args.verbose):
                # Indicate progress
                if args.verbose:
                    print("\nCreating visualizations...")
                else:
                    logger.info("Creating visualizations...")
                    
                # Create visualizations
                create_visualizations(analyses, cards_df, output_dir=args.output_dir, verbose=args.verbose)
                
                # Export results
                if args.verbose:
                    print("\nExporting detailed results to CSV...")
                else:
                    logger.info("Exporting detailed results to CSV...")
                    
                # Export to CSV
                csv_files = export_results_to_csv(all_results, output_dir=args.output_dir)
                
                # Show outputs only in verbose mode
                if csv_files and args.verbose:
                    print(f"CSV files created:")
                    for strategy, path in csv_files.items():
                        if strategy == 'all_strategies':
                            print(f"  - {os.path.basename(path)} (consolidated results)")
                        else:
                            print(f"  - {os.path.basename(path)}")
                
                # Final output
                if args.verbose:
                    print(f"\nAnalysis complete! Results saved in '{args.output_dir}' folder.")
                else:
                    logger.info(f"Analysis complete! Results saved in '{args.output_dir}' folder.")
        
        else:
            # Save partial results if interrupted
            logger.info("Simulation was interrupted. Saving partial analysis results...")
            
            # Process partial results with filtered output
            with OutputFilter(verbose=args.verbose):
                # Perform partial analysis on completed results
                for strategy, results in all_results.items():
                    if results:
                        logger.info(f"Analyzing {len(results)} results for {strategy}")
                        analysis = analyze_simulations(results, verbose=args.verbose)
                        analyses[strategy] = analysis
                
                # Save whatever we have
                save_partial_results(analyses, cards_df, args.output_dir)
                
                if len(analyses) > 0:
                    # Try to create visualizations for what we have
                    logger.info("Creating visualizations from partial results...")
                    try:
                        create_visualizations(analyses, cards_df, output_dir=args.output_dir, verbose=args.verbose)
                    except Exception as e:
                        logger.error(f"Error creating visualizations: {e}")
                    
                    # Export partial results to CSV
                    logger.info("Exporting partial results to CSV...")
                    try:
                        csv_files = export_results_to_csv(all_results, output_dir=args.output_dir)
                        if csv_files:
                            logger.info(f"Exported partial results to CSV files:")
                            for strategy, path in csv_files.items():
                                logger.info(f"  - {os.path.basename(path)}")
                    except Exception as e:
                        logger.error(f"Error exporting partial results to CSV: {e}")
        
        return analyses, cards_df
    
    except KeyboardInterrupt:
        logger.info("\nUser interrupted execution. Exiting.")
        return None, None
    
    except Exception as e:
        logger.exception(f"Error in main execution: {e}")
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    main()