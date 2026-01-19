"""Quick test to verify everything is working."""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.models.ollama_wrapper import OllamaModel
from src.guardrail_wrappers.guardrail_implementations import get_all_guardrails
from src.utils.dataset_loader import GuardrailTestDataset

print("Testing setup...\n")

# Test 1: Ollama connection
print("1. Testing Ollama connection...")
try:
    model = OllamaModel("llama3.2:3b")
    result = model.generate("Say 'Hello World'")
    if result['success']:
        print(f"   ✓ Ollama working! Response: {result['response'][:50]}...")
    else:
        print(f"   ✗ Ollama error: {result['error']}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 2: Guardrails
print("\n2. Testing Guardrails...")
try:
    guardrails = get_all_guardrails()
    print(f"   ✓ Loaded {len(guardrails)} guardrails:")
    for g in guardrails:
        print(f"      - {g.name}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Dataset
print("\n3. Testing Dataset Loader...")
try:
    loader = GuardrailTestDataset()
    prompts = loader.load_custom_prompts()
    print(f"   ✓ Loaded {len(prompts)} test prompts")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n✓ Setup test complete!")
print("\nTo run benchmark: python src/benchmarks/run_benchmark.py")
print("To visualize: python src/benchmarks/visualize_results.py")