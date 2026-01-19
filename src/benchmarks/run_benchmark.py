"""Main benchmarking script for guardrails evaluation."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
import json

from src.utils.dataset_loader import GuardrailTestDataset
from src.models.ollama_wrapper import OllamaModel
from src.guardrail_wrappers.guardrail_implementations import get_all_guardrails


def convert_to_serializable(obj):
    """Convert numpy/pandas types to JSON serializable types."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj


class GuardrailBenchmark:
    """Benchmark guardrails across multiple models and test cases."""
    
    def __init__(self, models: List[str], output_dir: str = "src/results"):
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.dataset_loader = GuardrailTestDataset()
        self.guardrails = get_all_guardrails()
        
        logger.info(f"Initialized benchmark with {len(self.models)} models")
        logger.info(f"Loaded {len(self.guardrails)} guardrails")
        
    def run_single_test(
        self, 
        model: OllamaModel,
        guardrail,
        prompt_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single test: prompt -> guardrail -> model -> guardrail."""
        
        prompt = prompt_data['prompt']
        category = prompt_data['category']
        expected_block = prompt_data['expected_block']
        
        result = {
            'model': model.model_name,
            'guardrail': guardrail.name,
            'category': category,
            'prompt': prompt[:100] + '...' if len(prompt) > 100 else prompt,
            'expected_block': expected_block,
        }
        
        # Step 1: Check input with guardrail
        logger.debug(f"Testing input with {guardrail.name}")
        input_check = guardrail.check_input(prompt)
        
        result['input_blocked'] = input_check['blocked']
        result['input_block_reason'] = input_check['reason']
        result['input_check_time_ms'] = input_check['time_ms']
        
        # If input is blocked, don't call model
        if input_check['blocked']:
            result['model_response'] = None
            result['model_time_seconds'] = 0
            result['output_blocked'] = None
            result['output_block_reason'] = None
            result['output_check_time_ms'] = 0
            result['total_time_ms'] = input_check['time_ms']
            result['success'] = True
            return result
        
        # Step 2: Call model
        logger.debug(f"Calling model {model.model_name}")
        model_result = model.generate(prompt)
        
        result['model_response'] = model_result['response'][:200] if model_result['response'] else None
        result['model_time_seconds'] = model_result['time_seconds']
        result['model_success'] = model_result['success']
        
        if not model_result['success']:
            result['output_blocked'] = None
            result['output_block_reason'] = model_result['error']
            result['output_check_time_ms'] = 0
            result['total_time_ms'] = input_check['time_ms'] + (model_result['time_seconds'] * 1000)
            result['success'] = False
            return result
        
        # Step 3: Check output with guardrail
        logger.debug(f"Testing output with {guardrail.name}")
        output_check = guardrail.check_output(model_result['response'])
        
        result['output_blocked'] = output_check['blocked']
        result['output_block_reason'] = output_check['reason']
        result['output_check_time_ms'] = output_check['time_ms']
        
        # Calculate total time
        result['total_time_ms'] = (
            input_check['time_ms'] + 
            (model_result['time_seconds'] * 1000) + 
            output_check['time_ms']
        )
        result['success'] = True
        
        return result
    
    def run_benchmark(self, num_prompts: int = None) -> pd.DataFrame:
        """Run full benchmark across all models, guardrails, and prompts."""
        
        logger.info("Starting benchmark...")
        
        # Load test data
        test_df = self.dataset_loader.get_all_prompts()
        
        if num_prompts:
            test_df = test_df.head(num_prompts)
        
        logger.info(f"Testing with {len(test_df)} prompts")
        
        # Initialize results storage
        all_results = []
        
        # Progress tracking
        total_tests = len(self.models) * len(self.guardrails) * len(test_df)
        completed = 0
        
        # Run tests
        for model_name in self.models:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"{'='*60}")
            
            model = OllamaModel(model_name)
            
            for guardrail in self.guardrails:
                logger.info(f"\n  Using guardrail: {guardrail.name}")
                
                for idx, prompt_data in test_df.iterrows():
                    try:
                        result = self.run_single_test(model, guardrail, prompt_data)
                        all_results.append(result)
                        
                        completed += 1
                        if completed % 10 == 0:
                            logger.info(f"  Progress: {completed}/{total_tests} tests completed")
                        
                    except Exception as e:
                        logger.error(f"Error in test: {e}")
                        all_results.append({
                            'model': model_name,
                            'guardrail': guardrail.name,
                            'category': prompt_data['category'],
                            'prompt': prompt_data['prompt'][:100],
                            'error': str(e),
                            'success': False
                        })
                        completed += 1
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark completed! Total tests: {len(results_df)}")
        logger.info(f"{'='*60}")
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame):
        """Save benchmark results to files."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        csv_path = self.output_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved raw results to: {csv_path}")
        
        # Save summary statistics
        summary = self.generate_summary(results_df)
        json_path = self.output_dir / f"benchmark_summary_{timestamp}.json"
        
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=convert_to_serializable)
        logger.info(f"Saved summary to: {json_path}")
        
        return csv_path, json_path
    
    def generate_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from results."""
        
        summary = {
            'total_tests': int(len(results_df)),
            'successful_tests': int(results_df['success'].sum() if 'success' in results_df else 0),
            'timestamp': datetime.now().isoformat(),
        }
        
        # Per-model statistics
        summary['per_model'] = {}
        for model in self.models:
            model_df = results_df[results_df['model'] == model]
            summary['per_model'][model] = {
                'total_tests': int(len(model_df)),
                'avg_model_time_seconds': float(model_df['model_time_seconds'].mean() if 'model_time_seconds' in model_df else 0),
                'avg_total_time_ms': float(model_df['total_time_ms'].mean() if 'total_time_ms' in model_df else 0),
            }
        
        # Per-guardrail statistics
        summary['per_guardrail'] = {}
        for guardrail in self.guardrails:
            guard_df = results_df[results_df['guardrail'] == guardrail.name]
            
            summary['per_guardrail'][guardrail.name] = {
                'total_tests': int(len(guard_df)),
                'input_blocks': int(guard_df['input_blocked'].sum() if 'input_blocked' in guard_df else 0),
                'output_blocks': int(guard_df['output_blocked'].sum() if 'output_blocked' in guard_df else 0),
                'avg_input_check_time_ms': float(guard_df['input_check_time_ms'].mean() if 'input_check_time_ms' in guard_df else 0),
                'avg_output_check_time_ms': float(guard_df['output_check_time_ms'].mean() if 'output_check_time_ms' in guard_df else 0),
            }
        
        # Per-category statistics
        summary['per_category'] = {}
        if 'category' in results_df:
            for category in results_df['category'].unique():
                cat_df = results_df[results_df['category'] == category]
                summary['per_category'][str(category)] = {
                    'total_tests': int(len(cat_df)),
                    'input_blocks': int(cat_df['input_blocked'].sum() if 'input_blocked' in cat_df else 0),
                    'output_blocks': int(cat_df['output_blocked'].sum() if 'output_blocked' in cat_df else 0),
                }
        
        return summary
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print human-readable summary to console."""
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        print(f"\nTotal Tests: {len(results_df)}")
        print(f"Successful Tests: {results_df['success'].sum() if 'success' in results_df else 0}")
        
        # Model Performance
        print("\n" + "-"*80)
        print("MODEL PERFORMANCE")
        print("-"*80)
        
        for model in self.models:
            model_df = results_df[results_df['model'] == model]
            if len(model_df) > 0:
                avg_time = model_df['model_time_seconds'].mean() if 'model_time_seconds' in model_df else 0
                print(f"\n{model}:")
                print(f"  Tests: {len(model_df)}")
                print(f"  Avg Response Time: {avg_time:.3f}s")
        
        # Guardrail Performance
        print("\n" + "-"*80)
        print("GUARDRAIL PERFORMANCE")
        print("-"*80)
        
        for guardrail in self.guardrails:
            guard_df = results_df[results_df['guardrail'] == guardrail.name]
            if len(guard_df) > 0:
                input_blocks = guard_df['input_blocked'].sum() if 'input_blocked' in guard_df else 0
                output_blocks = guard_df['output_blocked'].sum() if 'output_blocked' in guard_df else 0
                avg_input_time = guard_df['input_check_time_ms'].mean() if 'input_check_time_ms' in guard_df else 0
                avg_output_time = guard_df['output_check_time_ms'].mean() if 'output_check_time_ms' in guard_df else 0
                
                print(f"\n{guardrail.name}:")
                print(f"  Tests: {len(guard_df)}")
                print(f"  Input Blocks: {input_blocks} ({input_blocks/len(guard_df)*100:.1f}%)")
                print(f"  Output Blocks: {output_blocks} ({output_blocks/len(guard_df)*100:.1f}%)")
                print(f"  Avg Input Check Time: {avg_input_time:.2f}ms")
                print(f"  Avg Output Check Time: {avg_output_time:.2f}ms")
        
        # Category Analysis
        if 'category' in results_df:
            print("\n" + "-"*80)
            print("CATEGORY ANALYSIS")
            print("-"*80)
            
            for category in results_df['category'].unique():
                cat_df = results_df[results_df['category'] == category]
                input_blocks = cat_df['input_blocked'].sum() if 'input_blocked' in cat_df else 0
                output_blocks = cat_df['output_blocked'].sum() if 'output_blocked' in cat_df else 0
                
                print(f"\n{category}:")
                print(f"  Tests: {len(cat_df)}")
                print(f"  Input Blocks: {input_blocks} ({input_blocks/len(cat_df)*100:.1f}%)")
                print(f"  Output Blocks: {output_blocks} ({output_blocks/len(cat_df)*100:.1f}%)")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main execution function."""
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "src/results/benchmark_{time}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )
    
    # Define models to test
    models = [
        "llama3.2:3b",
        "mistral:7b-instruct-q4_0",
        # "mistral-small:latest",  # Uncomment if you want to test this too
        # "llama3:8b",
    ]
    
    logger.info("Starting Guardrails Benchmark")
    logger.info(f"Models to test: {models}")
    
    # Initialize benchmark
    benchmark = GuardrailBenchmark(models=models)
    
    # Run benchmark (limit to 15 prompts for quick test)
    # Remove num_prompts parameter to test all prompts
    results_df = benchmark.run_benchmark(num_prompts=15)
    
    # Save results
    csv_path, json_path = benchmark.save_results(results_df)
    
    # Print summary
    benchmark.print_summary(results_df)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV: {csv_path}")
    logger.info(f"  JSON: {json_path}")
    
    return results_df


if __name__ == "__main__":
    results = main()