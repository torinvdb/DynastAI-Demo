"""
Simulation runners for DynastAI with improved CTRL+C handling.
"""
import concurrent.futures
import logging
import signal
import sys
import time
import os
import multiprocessing
from tqdm import tqdm
from .engine import DynastAIGame
from .strategies import create_strategy
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Global variables to track simulation state
_shutdown_requested = False
_executors = []

# Simplified output filter for this module
class OutputFilter:
    """Context manager to filter out certain outputs when not in verbose mode"""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.original_stdout = None
        self.null_output = None
        
    def __enter__(self):
        if not self.verbose:
            self.original_stdout = sys.stdout
            self.null_output = open(os.devnull, 'w')
            sys.stdout = self.null_output
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.verbose:
            sys.stdout = self.original_stdout
            self.null_output.close()

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

def _init_worker():
    """
    Initialize worker process by ignoring SIGINT
    This ensures that only the main process handles the Ctrl+C
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def signal_handler(sig, frame):
    """
    Signal handler for graceful shutdown
    """
    global _shutdown_requested
    if not _shutdown_requested:
        logger.info("\nShutdown requested. Cleaning up workers... (Press Ctrl+C again to force exit)")
        _shutdown_requested = True
        
        # Shutdown all executors
        for executor in _executors:
            executor.shutdown(wait=False)
        
        # Allow clean exit on second Ctrl+C
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))

@quiet_unless_verbose
def run_single_simulation(cards_df, max_turns, strategy_name, verbose=False, **strategy_kwargs):
    """
    Run a single game simulation
    
    Parameters:
    -----------
    cards_df : pandas DataFrame
        DataFrame containing card data
    max_turns : int
        Maximum number of turns to play
    strategy_name : str
        Name of the strategy to use
    verbose : bool
        Whether to enable verbose output
    **strategy_kwargs : dict
        Additional arguments to pass to the strategy constructor
        
    Returns:
    --------
    dict
        Game statistics
    """
    # Check if shutdown was requested
    if _shutdown_requested:
        return None
        
    try:
        # Create the strategy
        strategy = create_strategy(strategy_name, **strategy_kwargs)
        
        # Create and run the game
        game = DynastAIGame(cards_df, strategy)
        return game.play_game(max_turns, verbose=verbose)
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        return None

def run_simulations(cards_df, num_sims=1000, max_turns=200, strategy_name='random', 
                    sim_workers=None, position=0, verbose=False, **strategy_kwargs):
    """
    Run multiple game simulations in parallel
    
    Parameters:
    -----------
    cards_df : pandas DataFrame
        DataFrame containing card data
    num_sims : int
        Number of simulations to run
    max_turns : int
        Maximum number of turns per game
    strategy_name : str
        Name of the strategy to use
    sim_workers : int or None
        Number of parallel workers to use (None for auto-selection)
    position : int
        Position for the progress bar
    verbose : bool
        Whether to enable verbose output
    **strategy_kwargs : dict
        Additional arguments to pass to the strategy constructor
        
    Returns:
    --------
    list
        List of game statistics dictionaries
    """
    global _executors, _shutdown_requested
    
    # Reset shutdown flag
    _shutdown_requested = False
    
    # Setup signal handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    
    results = []
    
    # Create a ProcessPoolExecutor with initializer that makes workers ignore SIGINT
    ctx = multiprocessing.get_context('spawn')  # Use spawn for better signal handling
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=sim_workers,
            mp_context=ctx,
            initializer=_init_worker) as executor:
            
        # Register executor for cleanup
        _executors.append(executor)
        
        # Create futures list
        futures = []
        for _ in range(num_sims):
            # Check if shutdown was requested before submitting more tasks
            if _shutdown_requested:
                break
                
            future = executor.submit(
                run_single_simulation,
                cards_df,
                max_turns,
                strategy_name,
                verbose=verbose,
                **strategy_kwargs
            )
            futures.append(future)
        
        try:
            # Use unique tqdm instance with clear positions to avoid corruption
            pbar = tqdm(
                total=num_sims, 
                desc=f"{strategy_name} simulations", 
                position=position, 
                leave=True,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            completed = 0
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                if _shutdown_requested:
                    # Don't wait for more results after shutdown request
                    break
                    
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error in simulation: {e}")
                
                completed += 1
                pbar.update(1)
                
                # Check cancellation after each update
                if _shutdown_requested:
                    pbar.write(f"Shutdown requested. Processed {completed}/{num_sims} simulations.")
                    break
            
            pbar.close()
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in main thread")
            _shutdown_requested = True
        
        finally:
            # Make sure we clear the executor
            if executor in _executors:
                _executors.remove(executor)
    
    # Restore original signal handler
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    if _shutdown_requested:
        logger.info(f"Simulation interrupted. Returning {len(results)} completed results.")
    
    return results

def run_strategy_simulations(strategy_name, cards_df, num_sims, max_turns, 
                            sim_workers, position, verbose=False, **strategy_kwargs):
    """
    Run simulations for a specific strategy and analyze the results
    
    Parameters:
    -----------
    strategy_name : str
        Name of the strategy to simulate
    cards_df : pandas DataFrame
        DataFrame containing card data
    num_sims : int
        Number of simulations to run
    max_turns : int
        Maximum number of turns per game
    sim_workers : int or None
        Number of parallel workers to use (None for auto-selection)
    position : int
        Position for the progress bar
    verbose : bool
        Whether to enable verbose output
    **strategy_kwargs : dict
        Additional arguments to pass to the strategy constructor
        
    Returns:
    --------
    tuple
        (strategy_name, simulation_results)
    """
    global _shutdown_requested
    
    logger.info(f"Running {num_sims} simulations with {strategy_name} strategy...")
    
    results = run_simulations(
        cards_df,
        num_sims=num_sims,
        max_turns=max_turns,
        strategy_name=strategy_name,
        sim_workers=sim_workers,
        position=position,
        verbose=verbose,
        **strategy_kwargs
    )
    
    if _shutdown_requested and not results:
        logger.warning(f"No results for {strategy_name} due to shutdown")
    
    return strategy_name, results

def run_multi_strategy_simulations(cards_df, strategies, num_sims=1000, max_turns=200, 
                                  strategy_workers=None, sim_workers=None, verbose=False):
    """
    Run simulations for multiple strategies in parallel
    
    Parameters:
    -----------
    cards_df : pandas DataFrame
        DataFrame containing card data
    strategies : list of str or dict
        List of strategy names to simulate, or dict mapping strategy names to kwargs
    num_sims : int
        Number of simulations to run per strategy
    max_turns : int
        Maximum number of turns per game
    strategy_workers : int or None
        Number of parallel strategy workers (None for auto-selection)
    sim_workers : int or None
        Number of parallel simulation workers per strategy (None for auto-selection)
    verbose : bool
        Whether to enable verbose output
        
    Returns:
    --------
    dict
        Dictionary mapping strategy names to lists of simulation results
    """
    global _executors, _shutdown_requested
    
    # Reset shutdown flag
    _shutdown_requested = False
    
    # Setup signal handler
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Convert strategies list to dict if necessary
    if isinstance(strategies, list):
        strategies = {strategy: {} for strategy in strategies}
    
    all_results = {}
    
    # Use sequential execution of strategies for better control
    # This sacrifices some parallelism but gives much better control over the processes
    # and more reliable cancellation
    for idx, (strategy_name, strategy_kwargs) in enumerate(strategies.items()):
        # Check if shutdown was requested before starting new strategy
        if _shutdown_requested:
            logger.info("Shutdown requested. Not starting new strategy simulations.")
            break
            
        try:
            strategy_name, results = run_strategy_simulations(
                strategy_name,
                cards_df,
                num_sims,
                max_turns,
                sim_workers,
                0,  # Always use position 0 since we're running sequentially
                verbose=verbose,
                **strategy_kwargs
            )
            all_results[strategy_name] = results
            
            # Print completion message
            logger.info(f"Completed {len(results)}/{num_sims} simulations for {strategy_name}")
            
        except Exception as e:
            logger.error(f"Error in strategy simulation for {strategy_name}: {e}")
            
        # Check if shutdown was requested after each strategy completes
        if _shutdown_requested:
            logger.info("Shutdown requested. Not starting new strategy simulations.")
            break
    
    # Restore original signal handler
    signal.signal(signal.SIGINT, original_sigint_handler)
    
    if _shutdown_requested:
        logger.info(f"Multi-strategy simulation interrupted. Returning results for {len(all_results)} completed strategies.")
    
    return all_results