"""Visualization script for benchmark results."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_latest_results(results_dir: str = "src/results") -> pd.DataFrame:
    """Load the most recent benchmark results."""
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("benchmark_results_*.csv"))
    
    if not csv_files:
        print("No results found!")
        return None
    
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    return pd.read_csv(latest_file)


def plot_guardrail_performance(df: pd.DataFrame, output_dir: str = "src/results"):
    """Plot guardrail blocking performance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Input blocks by guardrail
    input_blocks = df.groupby('guardrail')['input_blocked'].sum()
    axes[0, 0].bar(input_blocks.index, input_blocks.values, color='coral')
    axes[0, 0].set_title('Input Blocks by Guardrail', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Guardrail')
    axes[0, 0].set_ylabel('Number of Blocks')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Output blocks by guardrail
    output_blocks = df.groupby('guardrail')['output_blocked'].sum()
    axes[0, 1].bar(output_blocks.index, output_blocks.values, color='skyblue')
    axes[0, 1].set_title('Output Blocks by Guardrail', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Guardrail')
    axes[0, 1].set_ylabel('Number of Blocks')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Average input check time
    avg_input_time = df.groupby('guardrail')['input_check_time_ms'].mean()
    axes[1, 0].bar(avg_input_time.index, avg_input_time.values, color='lightgreen')
    axes[1, 0].set_title('Avg Input Check Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Guardrail')
    axes[1, 0].set_ylabel('Time (ms)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Average output check time
    avg_output_time = df.groupby('guardrail')['output_check_time_ms'].mean()
    axes[1, 1].bar(avg_output_time.index, avg_output_time.values, color='plum')
    axes[1, 1].set_title('Avg Output Check Time', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Guardrail')
    axes[1, 1].set_ylabel('Time (ms)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "guardrail_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved guardrail performance plot to: {output_path}")
    plt.close()


def plot_model_comparison(df: pd.DataFrame, output_dir: str = "src/results"):
    """Compare model performance."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Average response time by model
    avg_time = df.groupby('model')['model_time_seconds'].mean()
    axes[0].bar(avg_time.index, avg_time.values, color='teal')
    axes[0].set_title('Average Response Time by Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Time (seconds)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Total processing time (model + guardrails)
    avg_total_time = df.groupby('model')['total_time_ms'].mean() / 1000  # Convert to seconds
    axes[1].bar(avg_total_time.index, avg_total_time.values, color='orange')
    axes[1].set_title('Avg Total Time (Model + Guardrails)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison plot to: {output_path}")
    plt.close()


def plot_category_analysis(df: pd.DataFrame, output_dir: str = "src/results"):
    """Analyze blocking by category."""
    
    if 'category' not in df.columns:
        print("No category data available")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Blocks by category
    category_blocks = df.groupby('category')[['input_blocked', 'output_blocked']].sum()
    category_blocks.plot(kind='bar', ax=axes[0], color=['coral', 'skyblue'])
    axes[0].set_title('Blocks by Category', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Number of Blocks')
    axes[0].legend(['Input Blocks', 'Output Blocks'])
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Block rate by category
    category_total = df.groupby('category').size()
    input_block_rate = (df.groupby('category')['input_blocked'].sum() / category_total * 100)
    output_block_rate = (df.groupby('category')['output_blocked'].sum() / category_total * 100)
    
    x = range(len(input_block_rate))
    width = 0.35
    
    axes[1].bar([i - width/2 for i in x], input_block_rate.values, width, label='Input', color='coral')
    axes[1].bar([i + width/2 for i in x], output_block_rate.values, width, label='Output', color='skyblue')
    axes[1].set_title('Block Rate by Category (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Block Rate (%)')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(input_block_rate.index, rotation=45)
    axes[1].legend()
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "category_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved category analysis plot to: {output_path}")
    plt.close()


def plot_heatmap(df: pd.DataFrame, output_dir: str = "src/results"):
    """Create heatmap of blocks by model and guardrail."""
    
    # Create pivot table
    pivot_input = df.pivot_table(
        values='input_blocked',
        index='guardrail',
        columns='model',
        aggfunc='sum',
        fill_value=0
    )
    
    pivot_output = df.pivot_table(
        values='output_blocked',
        index='guardrail',
        columns='model',
        aggfunc='sum',
        fill_value=0
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Input blocks heatmap
    sns.heatmap(pivot_input, annot=True, fmt='g', cmap='Reds', ax=axes[0], cbar_kws={'label': 'Blocks'})
    axes[0].set_title('Input Blocks: Guardrail vs Model', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Guardrail')
    
    # Output blocks heatmap
    sns.heatmap(pivot_output, annot=True, fmt='g', cmap='Blues', ax=axes[1], cbar_kws={'label': 'Blocks'})
    axes[1].set_title('Output Blocks: Guardrail vs Model', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Guardrail')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "blocks_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_path}")
    plt.close()


def main():
    """Main visualization function."""
    
    print("Loading benchmark results...")
    df = load_latest_results()
    
    if df is None or len(df) == 0:
        print("No data to visualize!")
        return
    
    print(f"Loaded {len(df)} test results")
    print(f"Models: {df['model'].unique()}")
    print(f"Guardrails: {df['guardrail'].unique()}")
    
    print("\nGenerating visualizations...")
    
    plot_guardrail_performance(df)
    plot_model_comparison(df)
    plot_category_analysis(df)
    plot_heatmap(df)
    
    print("\nAll visualizations complete!")
    print("Check src/results/ directory for PNG files")


if __name__ == "__main__":
    main()