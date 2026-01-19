"""Detailed analysis of individual test cases."""

import pandas as pd
import glob
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def load_latest_results():
    """Load most recent results."""
    csv_files = glob.glob("src/results/benchmark_results_*.csv")
    if not csv_files:
        return None
    latest = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
    return pd.read_csv(latest)

def show_blocked_cases(df):
    """Show all blocked test cases."""
    console.print("\n[bold red]🔴 BLOCKED TEST CASES[/bold red]\n")
    
    blocked = df[df['input_blocked'] == True]
    
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Category", style="cyan")
    table.add_column("Guardrail", style="yellow")
    table.add_column("Prompt", style="white", max_width=50)
    table.add_column("Reason", style="red", max_width=40)
    
    for _, row in blocked.head(20).iterrows():
        table.add_row(
            row['category'],
            row['guardrail'],
            row['prompt'][:47] + "..." if len(row['prompt']) > 50 else row['prompt'],
            str(row['input_block_reason'])[:37] + "..." if pd.notna(row['input_block_reason']) and len(str(row['input_block_reason'])) > 40 else str(row['input_block_reason'])
        )
    
    console.print(table)
    console.print(f"\nTotal blocked: {len(blocked)}/{len(df)} ({len(blocked)/len(df)*100:.1f}%)\n")

def show_passed_dangerous(df):
    """Show dangerous prompts that passed through."""
    console.print("\n[bold yellow]⚠️  DANGEROUS PROMPTS THAT PASSED[/bold yellow]\n")
    
    dangerous_categories = ['toxic', 'pii', 'jailbreak', 'harmful', 'code_injection']
    dangerous = df[df['category'].isin(dangerous_categories)]
    passed = dangerous[dangerous['input_blocked'] == False]
    
    table = Table(show_header=True, header_style="bold yellow", box=box.ROUNDED)
    table.add_column("Category", style="red")
    table.add_column("Guardrail", style="cyan")
    table.add_column("Prompt", style="white", max_width=60)
    
    for _, row in passed.head(15).iterrows():
        table.add_row(
            row['category'],
            row['guardrail'],
            row['prompt'][:57] + "..." if len(row['prompt']) > 60 else row['prompt']
        )
    
    console.print(table)
    console.print(f"\n[bold red]⚠️  {len(passed)} dangerous prompts passed through![/bold red]\n")

def show_false_positives(df):
    """Show safe prompts that were blocked."""
    console.print("\n[bold cyan]🔵 FALSE POSITIVES (Safe prompts blocked)[/bold cyan]\n")
    
    safe = df[df['category'] == 'safe']
    false_pos = safe[safe['input_blocked'] == True]
    
    table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
    table.add_column("Guardrail", style="yellow")
    table.add_column("Prompt", style="white", max_width=60)
    table.add_column("Reason", style="red", max_width=40)
    
    for _, row in false_pos.iterrows():
        table.add_row(
            row['guardrail'],
            row['prompt'][:57] + "..." if len(row['prompt']) > 60 else row['prompt'],
            str(row['input_block_reason'])[:37] + "..." if pd.notna(row['input_block_reason']) else ""
        )
    
    console.print(table)
    console.print(f"\nFalse positive rate: {len(false_pos)}/{len(safe)} ({len(false_pos)/len(safe)*100:.1f}%)\n")

def compare_guardrails(df):
    """Compare guardrail effectiveness by category."""
    console.print("\n[bold green]📊 GUARDRAIL COMPARISON BY CATEGORY[/bold green]\n")
    
    categories = df['category'].unique()
    
    for category in categories:
        cat_df = df[df['category'] == category]
        
        table = Table(title=f"Category: {category.upper()}", show_header=True, header_style="bold", box=box.SIMPLE)
        table.add_column("Guardrail", style="cyan")
        table.add_column("Input Blocks", justify="right", style="red")
        table.add_column("Output Blocks", justify="right", style="yellow")
        table.add_column("Block Rate", justify="right", style="green")
        
        for guardrail in cat_df['guardrail'].unique():
            guard_df = cat_df[cat_df['guardrail'] == guardrail]
            input_blocks = guard_df['input_blocked'].sum()
            output_blocks = guard_df['output_blocked'].sum()
            block_rate = (input_blocks / len(guard_df) * 100) if len(guard_df) > 0 else 0
            
            table.add_row(
                guardrail,
                str(int(input_blocks)),
                str(int(output_blocks)),
                f"{block_rate:.1f}%"
            )
        
        console.print(table)
        console.print()

def main():
    console.print("\n[bold blue]🔍 DETAILED BENCHMARK ANALYSIS[/bold blue]\n")
    
    df = load_latest_results()
    if df is None:
        console.print("[red]No results found![/red]")
        return
    
    console.print(f"Loaded {len(df)} test results\n")
    
    # Show different analyses
    show_blocked_cases(df)
    show_passed_dangerous(df)
    show_false_positives(df)
    compare_guardrails(df)
    
    console.print("[bold green]✅ Analysis complete![/bold green]\n")

if __name__ == "__main__":
    main()