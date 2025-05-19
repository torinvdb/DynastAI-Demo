"""
Export functions for DynastAI simulation results.
"""
import os
import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def export_results_to_csv(all_results, output_dir='analysis_results'):
    """
    Export simulation results to CSV files for each strategy.
    
    Parameters:
    -----------
    all_results : dict
        Dictionary of strategy name to list of simulation results
    output_dir : str
        Directory to save CSV files
        
    Returns:
    --------
    dict
        Dictionary of strategy name to CSV file path
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store paths to exported CSV files
        csv_files = {}
        
        # Define fields for the consolidated CSV
        consolidated_fieldnames = [
            'strategy', 'run_id', 'turns_survived', 'end_reason',
            'final_piety', 'final_stability', 'final_power', 'final_wealth',
            'min_piety', 'min_stability', 'min_power', 'min_wealth',
            'max_piety', 'max_stability', 'max_power', 'max_wealth',
            'avg_piety', 'avg_stability', 'avg_power', 'avg_wealth',
            'std_piety', 'std_stability', 'std_power', 'std_wealth',
            'left_choices', 'right_choices', 'risky_card',
            'card_sequence', 'choice_sequence'
        ]
        
        # Create consolidated CSV file
        consolidated_csv_path = os.path.join(output_dir, "all_strategies_results.csv")
        consolidated_data = []
        
        # For debugging
        logger.info(f"Exporting {len(all_results)} strategies' results to CSV")
        logger.info(f"Found strategies: {', '.join(all_results.keys())}")
        
        # Check if any results are available
        total_results = sum(len(results) for results in all_results.values())
        if total_results == 0:
            logger.warning("No results to export")
            return {}
            
        logger.info(f"Total results to export: {total_results}")
        
        for strategy, results in all_results.items():
            logger.info(f"Strategy {strategy} has {len(results)} results")
            
        # Export each strategy's results to a separate CSV file
        for strategy, results in all_results.items():
            if not results:
                logger.warning(f"No results for {strategy} to export")
                continue
            
            # Define CSV file path
            csv_path = os.path.join(output_dir, f"{strategy}_results.csv")
            
            # Get field names from the first result
            # Flatten the nested structure for CSV export
            fieldnames = [
                'run_id', 'turns_survived', 'end_reason',
                'final_piety', 'final_stability', 'final_power', 'final_wealth',
                'min_piety', 'min_stability', 'min_power', 'min_wealth',
                'max_piety', 'max_stability', 'max_power', 'max_wealth',
                'avg_piety', 'avg_stability', 'avg_power', 'avg_wealth',
                'std_piety', 'std_stability', 'std_power', 'std_wealth',
                'left_choices', 'right_choices', 'risky_card',
                'card_sequence', 'choice_sequence'
            ]
            
            # Write to CSV
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Process each simulation result
                for i, result in enumerate(results):
                    try:
                        # Skip None results (failed simulations)
                        if result is None:
                            logger.warning(f"Skipping None result in strategy {strategy}")
                            continue
                        
                        # For debugging
                        logger.debug(f"Processing result {i+1} for strategy {strategy}")
                        if 'keys' in dir(result):
                            logger.debug(f"Available keys: {result.keys()}")
                        
                        # Extract the data we want to save
                        row = {
                            'run_id': i + 1,
                        }
                        
                        # Handle different field names and structures
                        if 'turns_survived' in result:
                            row['turns_survived'] = result['turns_survived']
                        elif 'turns' in result:
                            row['turns_survived'] = result['turns']
                        else:
                            row['turns_survived'] = 0
                            
                        row['end_reason'] = result.get('end_reason', 'Unknown')
                        
                        # Get final metrics differently based on how they're stored
                        metrics_history = []
                        final_metrics = {}
                        
                        # Try different ways the metrics might be stored
                        if 'metrics_history' in result and result['metrics_history'] is not None:
                            # If it's a list, use it directly
                            if isinstance(result['metrics_history'], list):
                                metrics_history = result['metrics_history']
                                if metrics_history:
                                    final_metrics = metrics_history[-1]
                            # If it's a DataFrame, convert to dict list
                            elif hasattr(result['metrics_history'], 'to_dict'):
                                try:
                                    metrics_df = result['metrics_history']
                                    metrics_records = metrics_df.to_dict('records')
                                    if metrics_records:
                                        metrics_history = metrics_records
                                        final_metrics = metrics_records[-1]
                                except Exception as e:
                                    logger.error(f"Error converting metrics history DataFrame: {e}")
                        
                        # If no metrics_history, try final_metrics
                        if not metrics_history and 'final_metrics' in result:
                            final_metrics = result['final_metrics']
                            metrics_history = [final_metrics]
                        
                        # If still no metrics, check for metrics in history
                        if not metrics_history and 'history' in result:
                            try:
                                history = result['history']
                                if hasattr(history, 'to_dict'):
                                    available_columns = history.columns.tolist()
                                    logger.debug(f"History columns: {available_columns}")
                                    history_records = history.to_dict('records')
                                    if history_records:
                                        # Extract metrics from history if available
                                        metrics_columns = ['Piety', 'Stability', 'Power', 'Wealth']
                                        
                                        # Check if we should look for lowercase column names
                                        if not all(col in available_columns for col in metrics_columns):
                                            lowercase_metrics = [col.lower() for col in metrics_columns]
                                            if all(col in [c.lower() for c in available_columns] for col in lowercase_metrics):
                                                # Map available columns to expected columns (case-insensitive)
                                                column_map = {}
                                                for expected in metrics_columns:
                                                    for available in available_columns:
                                                        if expected.lower() == available.lower():
                                                            column_map[expected] = available
                                                
                                                # Use the column map
                                                for record in history_records:
                                                    metrics_entry = {}
                                                    for expected, actual in column_map.items():
                                                        if actual in record:
                                                            metrics_entry[expected] = record[actual]
                                                    
                                                    if len(metrics_entry) == len(metrics_columns):
                                                        metrics_history.append(metrics_entry)
                                            
                                        else:
                                            # Use the exact column names
                                            for record in history_records:
                                                if all(metric in record for metric in metrics_columns):
                                                    metrics_entry = {metric: record[metric] for metric in metrics_columns}
                                                    metrics_history.append(metrics_entry)
                                        
                                        if metrics_history:
                                            final_metrics = metrics_history[-1]
                            except Exception as e:
                                logger.error(f"Error processing history for metrics: {e}")
                        
                        # If still no metrics, use default values
                        if not final_metrics:
                            logger.warning(f"No metrics found in result {i+1} for {strategy}")
                            final_metrics = {'Piety': 0, 'Stability': 0, 'Power': 0, 'Wealth': 0}
                            metrics_history = [final_metrics]
                        
                        # Add final metrics
                        row['final_piety'] = final_metrics.get('Piety', 0)
                        row['final_stability'] = final_metrics.get('Stability', 0)
                        row['final_power'] = final_metrics.get('Power', 0)
                        row['final_wealth'] = final_metrics.get('Wealth', 0)
                        
                        # Calculate min, max, avg, std for each metric
                        metrics = {'Piety': [], 'Stability': [], 'Power': [], 'Wealth': []}
                        for state in metrics_history:
                            for metric in metrics:
                                if metric in state:
                                    try:
                                        # Try to convert to float first, in case it's a string or other type
                                        metrics[metric].append(float(state[metric]))
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not convert metric {metric} value to float")
                        
                        # Add min, max, avg values
                        for metric, values in metrics.items():
                            metric_lower = metric.lower()
                            if values:  # Only calculate if we have values
                                try:
                                    row[f'min_{metric_lower}'] = min(values)
                                    row[f'max_{metric_lower}'] = max(values)
                                    row[f'avg_{metric_lower}'] = sum(values) / len(values)
                                    
                                    # Calculate standard deviation
                                    mean = sum(values) / len(values)
                                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                                    row[f'std_{metric_lower}'] = variance ** 0.5
                                except Exception as e:
                                    logger.error(f"Error calculating stats for {metric}: {e}")
                                    row[f'min_{metric_lower}'] = 0
                                    row[f'max_{metric_lower}'] = 0
                                    row[f'avg_{metric_lower}'] = 0
                                    row[f'std_{metric_lower}'] = 0
                            else:
                                # Default values if no data
                                row[f'min_{metric_lower}'] = 0
                                row[f'max_{metric_lower}'] = 0
                                row[f'avg_{metric_lower}'] = 0
                                row[f'std_{metric_lower}'] = 0
                        
                        # Extract card choice data
                        # First try common fields
                        choices = []
                        if 'choices' in result:
                            choices = result['choices']
                        elif 'choice_history' in result:
                            choices = result['choice_history']
                                
                        # Count choices
                        if isinstance(choices, list):
                            row['left_choices'] = choices.count('Left')
                            row['right_choices'] = choices.count('Right')
                        else:
                            row['left_choices'] = 0
                            row['right_choices'] = 0
                        
                        # Add risky card if available
                        if 'risky_card' in result:
                            row['risky_card'] = str(result['risky_card'])
                        else:
                            row['risky_card'] = ''
                        
                        # Add card sequence and choice sequence
                        card_ids = []
                        if 'card_ids' in result:
                            card_ids = result['card_ids']
                        elif 'played_cards' in result:
                            card_ids = result['played_cards']
                        elif 'history' in result and hasattr(result['history'], 'to_dict'):
                            try:
                                # Check what columns are available in the history DataFrame
                                available_columns = result['history'].columns.tolist()
                                logger.debug(f"Available history columns: {available_columns}")
                                
                                # Extract card IDs from history based on available columns
                                history_dict = result['history'].to_dict('records')
                                
                                # Try different column names for card ID
                                card_id_cols = ['Card_ID', 'card_id', 'CardID', 'Card']
                                for col in card_id_cols:
                                    if col in available_columns:
                                        card_ids = [str(entry.get(col, '')) for entry in history_dict if col in entry and entry.get(col) is not None]
                                        if card_ids:
                                            logger.debug(f"Found {len(card_ids)} card IDs in column {col}")
                                            break
                                
                                # If no choice field found yet, try to extract it from history
                                if not choices and history_dict:
                                    choice_cols = ['Choice', 'choice', 'Decision', 'decision']
                                    for col in choice_cols:
                                        if col in available_columns:
                                            choices = [str(entry.get(col, '')) for entry in history_dict if col in entry and entry.get(col) is not None]
                                            if choices:
                                                logger.debug(f"Found {len(choices)} choices in column {col}")
                                                break
                            except Exception as e:
                                logger.error(f"Error extracting data from history: {e}")
                        
                        # Format the sequence fields - ensure all entries are strings
                        if card_ids:
                            # Clean up the card_ids to ensure they're valid for CSV
                            clean_card_ids = []
                            for card_id in card_ids:
                                try:
                                    # Convert to string safely
                                    clean_card_ids.append(str(card_id).strip())
                                except:
                                    clean_card_ids.append("")
                            row['card_sequence'] = ','.join(clean_card_ids)
                        else:
                            row['card_sequence'] = ''
                            
                        if choices:
                            # Clean up the choices to ensure they're valid for CSV
                            clean_choices = []
                            for choice in choices:
                                try:
                                    # Convert to string safely
                                    clean_choices.append(str(choice).strip())
                                except:
                                    clean_choices.append("")
                            row['choice_sequence'] = ','.join(clean_choices)
                        else:
                            row['choice_sequence'] = ''
                        
                        # Write row to CSV
                        writer.writerow(row)
                        
                        # Add to consolidated data
                        consolidated_row = {'strategy': strategy}
                        consolidated_row.update(row)
                        consolidated_data.append(consolidated_row)
                        
                    except Exception as e:
                        logger.error(f"Error processing result {i+1} for strategy {strategy}: {e}")
                        continue
            
            csv_files[strategy] = csv_path
            logger.info(f"Exported {len(results)} results for strategy {strategy} to {csv_path}")
        
        # Write consolidated CSV
        if consolidated_data:
            with open(consolidated_csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=consolidated_fieldnames)
                writer.writeheader()
                for row in consolidated_data:
                    writer.writerow(row)
            
            csv_files['all_strategies'] = consolidated_csv_path
            logger.info(f"Exported consolidated results to {consolidated_csv_path}")
        
        return csv_files
    
    except Exception as e:
        logger.error(f"Error exporting results to CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}