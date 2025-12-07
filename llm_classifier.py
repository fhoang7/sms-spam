#%% Imports
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from anthropic import Anthropic
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import print_evaluation

#%% Configuration
print("="*70)
print("LLM-BASED SPAM CLASSIFICATION (BATCH API)")
print("="*70)

DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DIR, 'data')
GRAPHS_DIR = os.path.join(DIR, 'graphs')

# Create graphs directory if needed
if not os.path.exists(GRAPHS_DIR):
    os.makedirs(GRAPHS_DIR)

# Cost safety configuration
MAX_BUDGET = 20.0  # Maximum $20 budget across ALL API calls
SAFETY_MARGIN = 1.15  # 15% safety margin for cost estimation

# Anthropic API pricing (per million tokens) - HAIKU ONLY
# https://www.anthropic.com/pricing
# Batch API gets 50% discount on base prices
MODEL_NAME = 'claude-3-5-haiku-20241022'
PRICING = {
    'input': 0.25 * 0.5,    # $0.125 per MTok (50% batch discount)
    'output': 1.25 * 0.5    # $0.625 per MTok (50% batch discount)
}

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Max Budget: ${MAX_BUDGET:.2f}")
print(f"  Input Cost: ${PRICING['input']:.3f} per MTok (Batch API)")
print(f"  Output Cost: ${PRICING['output']:.3f} per MTok (Batch API)")

#%% Budget Tracking File
budget_file = os.path.join(GRAPHS_DIR, 'llm_budget_tracker.json')

def load_budget_tracker():
    """Load the budget tracker to track total spend across all runs."""
    if os.path.exists(budget_file):
        with open(budget_file, 'r') as f:
            tracker = json.load(f)
        print(f"\n✓ Budget tracker loaded")
        print(f"  Total spent so far: ${tracker['total_spent']:.4f}")
        print(f"  Remaining budget: ${MAX_BUDGET - tracker['total_spent']:.4f}")
        return tracker
    else:
        return {
            'total_spent': 0.0,
            'runs': []
        }

def save_budget_tracker(tracker):
    """Save the budget tracker."""
    with open(budget_file, 'w') as f:
        json.dump(tracker, f, indent=2)

def check_budget_available(tracker, estimated_cost):
    """Check if we have budget available for estimated cost."""
    remaining = MAX_BUDGET - tracker['total_spent']
    if estimated_cost > remaining:
        print(f"\n⚠ INSUFFICIENT BUDGET!")
        print(f"  Total spent: ${tracker['total_spent']:.4f}")
        print(f"  Remaining: ${remaining:.4f}")
        print(f"  Estimated need: ${estimated_cost:.4f}")
        print(f"  Shortfall: ${estimated_cost - remaining:.4f}")
        return False
    return True

#%% Load API Key
print("\n" + "="*70)
print("API AUTHENTICATION")
print("="*70)

api_key = os.environ.get('ANTHROPIC_API_KEY')
if not api_key:
    print("\n✗ ERROR: ANTHROPIC_API_KEY environment variable not set")
    print("\nPlease set your API key:")
    print("  export ANTHROPIC_API_KEY=your_api_key_here")
    print("\nGet your API key from: https://console.anthropic.com/")
    exit(1)

print("✓ API key found")

# Initialize Anthropic client
client = Anthropic(api_key=api_key)
print("✓ Anthropic client initialized")

# Load budget tracker
budget_tracker = load_budget_tracker()

#%% Load Test Data
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

X_test = pd.read_csv(os.path.join(DB_PATH, 'X_test.csv'))['message']
y_test = np.load(os.path.join(DB_PATH, 'y_test.npy'))

print(f"✓ Loaded {len(X_test)} test messages")
print(f"  Spam: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")
print(f"  Ham: {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")

#%% Token Estimation Function
def estimate_tokens(text):
    """
    Estimate token count for a text string.
    Uses conservative estimate: ~4 characters per token for English text.
    """
    return len(text) // 4 + 10  # +10 for safety margin

def estimate_cost(input_tokens, output_tokens):
    """Calculate estimated cost based on token counts."""
    input_cost = (input_tokens / 1_000_000) * PRICING['input']
    output_cost = (output_tokens / 1_000_000) * PRICING['output']
    return input_cost + output_cost

#%% Create Prompt Template
def create_prompt(message):
    """Create a simple binary classification prompt."""
    return f"""Is this SMS message spam?

Message: {message}

Answer only "yes" or "no"."""

#%% Load or Create Results Cache
cache_file = os.path.join(GRAPHS_DIR, 'llm_results_cache.json')

if os.path.exists(cache_file):
    print("\n" + "="*70)
    print("LOADING CACHED RESULTS")
    print("="*70)

    with open(cache_file, 'r') as f:
        cache = json.load(f)

    print(f"✓ Loaded {len(cache['results'])} cached predictions")
    print(f"  Continuing from where we left off...")

    results = cache['results']
    processed_indices = set(r['index'] for r in results)
    start_idx = 0
else:
    print("\n" + "="*70)
    print("STARTING FRESH CLASSIFICATION")
    print("="*70)

    results = []
    processed_indices = set()
    start_idx = 0

#%% Prepare Batch Requests
print("\n" + "="*70)
print("PREPARING BATCH REQUESTS")
print("="*70)

# Find messages that haven't been processed yet
remaining_messages = []
for idx in range(len(X_test)):
    if idx not in processed_indices:
        remaining_messages.append({
            'index': idx,
            'message': X_test.iloc[idx],
            'true_label': int(y_test[idx])
        })

print(f"\nMessages to process: {len(remaining_messages)}")

if len(remaining_messages) == 0:
    print("✓ All messages already processed!")
else:
    # Estimate total cost for remaining messages
    total_estimated_input_tokens = 0
    total_estimated_output_tokens = 0

    for msg_data in remaining_messages:
        prompt = create_prompt(msg_data['message'])
        total_estimated_input_tokens += estimate_tokens(prompt)
        total_estimated_output_tokens += 10  # "yes" or "no" estimate

    estimated_total_cost = estimate_cost(
        total_estimated_input_tokens,
        total_estimated_output_tokens
    ) * SAFETY_MARGIN

    print(f"\nCost Estimation:")
    print(f"  Estimated input tokens: {total_estimated_input_tokens:,}")
    print(f"  Estimated output tokens: {total_estimated_output_tokens:,}")
    print(f"  Estimated cost (with {SAFETY_MARGIN-1:.0%} margin): ${estimated_total_cost:.4f}")

    # Check if we have budget
    if not check_budget_available(budget_tracker, estimated_total_cost):
        # Calculate how many messages we can afford
        avg_cost_per_msg = estimated_total_cost / len(remaining_messages)
        remaining_budget = MAX_BUDGET - budget_tracker['total_spent']
        affordable_count = int(remaining_budget / avg_cost_per_msg)

        if affordable_count == 0:
            print(f"\n✗ Cannot process any more messages within budget.")
            print(f"  Total budget: ${MAX_BUDGET:.2f}")
            print(f"  Already spent: ${budget_tracker['total_spent']:.4f}")
            print(f"  No remaining budget for batch processing.")
            exit(0)

        print(f"\n⚠ Limiting to {affordable_count} messages to stay within budget")
        remaining_messages = remaining_messages[:affordable_count]

        # Recalculate estimate
        total_estimated_input_tokens = 0
        total_estimated_output_tokens = 0
        for msg_data in remaining_messages:
            prompt = create_prompt(msg_data['message'])
            total_estimated_input_tokens += estimate_tokens(prompt)
            total_estimated_output_tokens += 10

        estimated_total_cost = estimate_cost(
            total_estimated_input_tokens,
            total_estimated_output_tokens
        ) * SAFETY_MARGIN

        print(f"  Revised estimate: ${estimated_total_cost:.4f}")

    # Create batch request format
    print(f"\n" + "="*70)
    print("CREATING BATCH API REQUEST")
    print("="*70)

    batch_requests = []
    for msg_data in remaining_messages:
        prompt = create_prompt(msg_data['message'])

        batch_requests.append({
            "custom_id": f"msg_{msg_data['index']}",
            "params": {
                "model": MODEL_NAME,
                "max_tokens": 10,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        })

    print(f"✓ Created {len(batch_requests)} batch requests")

    # Save batch requests to file (required for Batch API)
    batch_requests_file = os.path.join(GRAPHS_DIR, 'batch_requests.jsonl')
    with open(batch_requests_file, 'w') as f:
        for req in batch_requests:
            f.write(json.dumps(req) + '\n')

    print(f"✓ Saved batch requests to {batch_requests_file}")

    #%% Submit Batch via API
    print("\n" + "="*70)
    print("SUBMITTING BATCH TO ANTHROPIC API")
    print("="*70)

    try:
        # Create the batch
        print("Submitting batch...")
        batch_start_time = time.time()

        message_batch = client.messages.batches.create(
            requests=batch_requests
        )

        print(f"✓ Batch created successfully!")
        print(f"  Batch ID: {message_batch.id}")
        print(f"  Status: {message_batch.processing_status}")
        print(f"  Request counts: {message_batch.request_counts}")

        # Poll for completion
        print(f"\nWaiting for batch to process...")
        print("  (This may take several minutes...)")

        while message_batch.processing_status in ['in_progress', 'pending']:
            time.sleep(10)  # Poll every 10 seconds

            # Retrieve batch status
            message_batch = client.messages.batches.retrieve(message_batch.id)

            elapsed = time.time() - batch_start_time
            print(f"  Status: {message_batch.processing_status} | "
                  f"Elapsed: {elapsed:.0f}s | "
                  f"Counts: {message_batch.request_counts}")

        batch_total_time = time.time() - batch_start_time

        print(f"\n✓ Batch processing complete!")
        print(f"  Final status: {message_batch.processing_status}")
        print(f"  Total time: {batch_total_time:.1f}s")
        print(f"  Request counts: {message_batch.request_counts}")

        # Check if batch succeeded
        if message_batch.processing_status != 'ended':
            print(f"\n✗ Batch processing failed with status: {message_batch.processing_status}")
            exit(1)

        #%% Retrieve and Process Results
        print("\n" + "="*70)
        print("RETRIEVING BATCH RESULTS")
        print("="*70)

        # Get results iterator
        batch_results = client.messages.batches.results(message_batch.id)

        # Process results
        actual_total_cost = 0.0
        new_results = []

        print("Processing results...")
        for result in batch_results:
            # Extract custom_id to get message index
            custom_id = result.custom_id
            msg_idx = int(custom_id.split('_')[1])

            # Find the original message data
            msg_data = next(m for m in remaining_messages if m['index'] == msg_idx)

            if result.result.type == 'succeeded':
                # Extract response
                message_result = result.result.message
                response_text = message_result.content[0].text.strip().lower()

                # Get token usage
                input_tokens = message_result.usage.input_tokens
                output_tokens = message_result.usage.output_tokens

                # Calculate cost
                cost = estimate_cost(input_tokens, output_tokens)
                actual_total_cost += cost

                # Parse yes/no to 1/0
                if 'yes' in response_text:
                    prediction = 1  # Spam
                elif 'no' in response_text:
                    prediction = 0  # Ham
                else:
                    print(f"  ⚠ Unexpected response for msg {msg_idx}: '{response_text}' - defaulting to spam")
                    prediction = 1

                # Store result
                new_results.append({
                    'index': msg_idx,
                    'message': msg_data['message'],
                    'true_label': msg_data['true_label'],
                    'predicted_label': prediction,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'cost': cost,
                    'response_text': response_text
                })
            else:
                # Handle errors
                print(f"  ✗ Error processing msg {msg_idx}: {result.result.type}")

        print(f"✓ Processed {len(new_results)} results")
        print(f"  Actual total cost: ${actual_total_cost:.4f}")

        # Update budget tracker
        budget_tracker['total_spent'] += actual_total_cost
        budget_tracker['runs'].append({
            'timestamp': datetime.now().isoformat(),
            'batch_id': message_batch.id,
            'messages_processed': len(new_results),
            'cost': actual_total_cost,
            'total_spent_after': budget_tracker['total_spent']
        })
        save_budget_tracker(budget_tracker)

        print(f"\n💰 Budget Update:")
        print(f"  This batch: ${actual_total_cost:.4f}")
        print(f"  Total spent: ${budget_tracker['total_spent']:.4f}")
        print(f"  Remaining: ${MAX_BUDGET - budget_tracker['total_spent']:.4f}")

        # Merge with existing results
        results.extend(new_results)

        # Save updated cache
        cache_data = {
            'model': MODEL_NAME,
            'timestamp': datetime.now().isoformat(),
            'total_messages': len(X_test),
            'processed_messages': len(results),
            'results': results
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"✓ Saved cache with {len(results)} total results")

    except Exception as e:
        print(f"\n✗ Batch API Error: {e}")
        print("\nSaving any partial results...")
        if results:
            cache_data = {
                'model': MODEL_NAME,
                'timestamp': datetime.now().isoformat(),
                'total_messages': len(X_test),
                'processed_messages': len(results),
                'results': results
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        exit(1)

#%% Performance Evaluation
print("\n" + "="*70)
print("PERFORMANCE EVALUATION")
print("="*70)

if len(results) == 0:
    print("No results to evaluate!")
    exit(0)

# Extract predictions and labels
y_true = [r['true_label'] for r in results]
y_pred = [r['predicted_label'] for r in results]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

# Detailed evaluation
print_evaluation(y_true, y_pred, title=f"LLM Classifier ({MODEL_NAME})")

#%% Cost Analysis
print("\n" + "="*70)
print("COST ANALYSIS")
print("="*70)

total_input_tokens = sum(r['input_tokens'] for r in results)
total_output_tokens = sum(r['output_tokens'] for r in results)
total_tokens = total_input_tokens + total_output_tokens

avg_input_tokens = total_input_tokens / len(results)
avg_output_tokens = total_output_tokens / len(results)

# Calculate total cost from results
total_result_cost = sum(r['cost'] for r in results)
avg_cost_per_message = total_result_cost / len(results)

print(f"\nToken Usage:")
print(f"  Total input tokens:  {total_input_tokens:,}")
print(f"  Total output tokens: {total_output_tokens:,}")
print(f"  Total tokens:        {total_tokens:,}")
print(f"  Avg input/message:   {avg_input_tokens:.1f}")
print(f"  Avg output/message:  {avg_output_tokens:.1f}")

print(f"\nCost Breakdown:")
print(f"  Total cost (from budget tracker): ${budget_tracker['total_spent']:.4f}")
print(f"  Avg cost per message: ${avg_cost_per_message:.6f}")
print(f"  Messages processed:  {len(results):,}")

print(f"\nCost Projections:")
print(f"  Cost per 1K messages:   ${avg_cost_per_message * 1_000:.2f}")
print(f"  Cost per 10K messages:  ${avg_cost_per_message * 10_000:.2f}")
print(f"  Cost per 100K messages: ${avg_cost_per_message * 100_000:.2f}")
print(f"  Cost per 1M messages:   ${avg_cost_per_message * 1_000_000:.2f}")

#%% Save Detailed Results
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save results as CSV for easy analysis
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(GRAPHS_DIR, 'llm_classifier_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"✓ Saved detailed results to {results_csv_path}")

# Save summary metrics
summary = {
    'model': MODEL_NAME,
    'timestamp': datetime.now().isoformat(),
    'messages_processed': len(results),
    'total_messages': len(X_test),
    'coverage_pct': (len(results) / len(X_test)) * 100,

    # Performance
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,

    # Cost
    'total_cost': budget_tracker['total_spent'],
    'avg_cost_per_message': avg_cost_per_message,
    'total_input_tokens': total_input_tokens,
    'total_output_tokens': total_output_tokens,
    'avg_input_tokens': avg_input_tokens,
    'avg_output_tokens': avg_output_tokens,
    'cost_per_1k': avg_cost_per_message * 1000,
    'cost_per_1m': avg_cost_per_message * 1_000_000,

    # Pricing
    'input_price_per_mtok': PRICING['input'],
    'output_price_per_mtok': PRICING['output'],
    'batch_api_discount': 0.5
}

summary_path = os.path.join(GRAPHS_DIR, 'llm_performance_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Saved performance summary to {summary_path}")

#%% Generate Visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Cost per Message Distribution
ax1 = axes[0, 0]
costs = [r['cost'] for r in results]
ax1.hist(costs, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
ax1.axvline(avg_cost_per_message, color='red', linestyle='--', linewidth=2,
            label=f'Mean: ${avg_cost_per_message:.6f}')
ax1.set_xlabel('Cost per Message ($)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Cost Distribution per Message (Batch API)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Token Usage
ax2 = axes[0, 1]
input_tokens = [r['input_tokens'] for r in results]
output_tokens = [r['output_tokens'] for r in results]

x = np.arange(2)
width = 0.35
means = [np.mean(input_tokens), np.mean(output_tokens)]
stds = [np.std(input_tokens), np.std(output_tokens)]

bars = ax2.bar(x, means, width, yerr=stds, color=['steelblue', 'forestgreen'],
               alpha=0.7, capsize=5, edgecolor='black')
ax2.set_ylabel('Tokens', fontsize=12, fontweight='bold')
ax2.set_title('Average Token Usage per Message', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Input Tokens', 'Output Tokens'], fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{mean:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Budget Tracking
ax3 = axes[1, 0]
cumulative_cost = np.cumsum([r['cost'] for r in results])
ax3.plot(range(len(cumulative_cost)), cumulative_cost, color='purple', linewidth=2)
ax3.axhline(MAX_BUDGET, color='red', linestyle='--', linewidth=2, label=f'Budget Limit: ${MAX_BUDGET}')
ax3.axhline(budget_tracker['total_spent'], color='green', linestyle='--', linewidth=2,
            label=f'Total Spent: ${budget_tracker["total_spent"]:.4f}')
ax3.set_xlabel('Messages Processed', fontsize=12, fontweight='bold')
ax3.set_ylabel('Cumulative Cost ($)', fontsize=12, fontweight='bold')
ax3.set_title('Cumulative Cost Over Messages', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Performance Metrics
ax4 = axes[1, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['steelblue', 'forestgreen', 'orange', 'purple']

bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('LLM Classifier Performance Metrics', fontsize=14, fontweight='bold')
ax4.set_ylim([0, 1.05])
ax4.grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
viz_path = os.path.join(GRAPHS_DIR, 'llm_cost_performance_analysis.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to {viz_path}")
plt.close()

#%% Final Summary
print("\n" + "="*70)
print("LLM CLASSIFICATION COMPLETE")
print("="*70)

print(f"\n📊 Performance:")
print(f"   Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

print(f"\n💰 Cost:")
print(f"   Total: ${budget_tracker['total_spent']:.4f} | Per message: ${avg_cost_per_message:.6f} | Per 1K: ${avg_cost_per_message * 1000:.2f}")

print(f"\n📁 Output files:")
print(f"   - {results_csv_path}")
print(f"   - {summary_path}")
print(f"   - {viz_path}")
print(f"   - {budget_file}")

if len(results) < len(X_test):
    remaining = len(X_test) - len(results)
    print(f"\n⚠ Note: Processed {len(results)}/{len(X_test)} messages ({len(results)/len(X_test)*100:.1f}%)")
    print(f"  {remaining} messages remaining")
    print(f"  Budget remaining: ${MAX_BUDGET - budget_tracker['total_spent']:.4f}")
    if MAX_BUDGET - budget_tracker['total_spent'] > 0:
        print(f"  Re-run this script to process more messages within remaining budget.")
else:
    print(f"\n✓ Processed all {len(X_test)} test messages!")

print("\n" + "="*70 + "\n")

# %%
