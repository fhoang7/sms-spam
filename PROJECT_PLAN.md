# SMS Spam Detection - Extended Project Plan

## Overview
This project compares multiple approaches to SMS spam classification and develops an optimized hybrid system that balances accuracy, cost, and speed.

## Phase 1: Baseline Approaches ✅ COMPLETED

### 1.1 Traditional ML (TF-IDF) ✅
**File:** `traditional_ml.py`
- TF-IDF vectorization (3000 features, unigrams + bigrams)
- Tested 30+ classical ML algorithms using LazyPredict
- Performance metrics: Accuracy, Precision, Recall, F1, ROC AUC
- **Output:** `graphs/traditional_ml_results.csv`, visualizations

### 1.2 Embedding-based ML ✅
**File:** `embedding_classifier.py`
- Sentence transformers (all-MiniLM-L6-v2, 384D embeddings)
- Same 30+ classifiers tested on semantic embeddings
- ChromaDB for vector storage and semantic search
- **Output:** `graphs/embedding_ml_results.csv`, visualizations

### 1.3 Baseline Comparison ✅
**File:** `comparison.py`
- Side-by-side performance comparison
- Statistical analysis and improvement metrics
- Multiple visualizations (bar charts, distributions, heatmaps)
- **Output:** `graphs/comparison_report.txt`, comparative charts

---

## Phase 2: LLM Classification ⏳ IN PROGRESS

### 2.1 Direct LLM Testing 🔄 NEXT
**File:** `llm_classifier.py` (to be created)

**Goals:**
- Test Anthropic API (Claude) for binary spam classification
- Use simple prompt: "Is this message spam? Answer yes or no: {message}"
- Measure key metrics for entire test set

**Metrics to Track:**
1. **Performance:**
   - Accuracy, Precision, Recall, F1-Score
   - Compare with Traditional ML and Embedding approaches

2. **Cost Analysis:**
   - Input tokens per message (prompt + message)
   - Output tokens per response (yes/no)
   - Cost per message (based on Anthropic pricing)
   - Total cost for test set (~1,114 messages)
   - Projected cost per 1M messages

3. **Speed Analysis:**
   - Latency per API call (p50, p95, p99)
   - Throughput (messages per second)
   - Compare with local ML inference time

**Implementation Details:**
- Use Claude 3.5 Haiku (fast, cheap) as baseline
- Also test Claude 3.5 Sonnet for comparison
- Batch API calls efficiently (respect rate limits)
- Log all requests for cost analysis
- Cache results to avoid re-running
- Handle API errors and retries gracefully

**Expected Outputs:**
```
graphs/
├── llm_performance.json         # Accuracy metrics
├── llm_cost_analysis.json       # Cost breakdown
├── llm_latency_distribution.png # Speed analysis
└── llm_vs_ml_comparison.csv     # Head-to-head comparison
```

---

## Phase 3: Hybrid Cascade System ⏳ PLANNED

### 3.1 Optimal Threshold Finding 🔄
**File:** `threshold_optimizer.py` (to be created)

**Goals:**
- Find TF-IDF probability threshold that achieves 100% spam recall
- Minimize false positives while maintaining perfect spam detection
- Analyze precision/recall trade-offs at different thresholds

**Approach:**
1. Train best TF-IDF model from Phase 1
2. Get prediction probabilities on test set
3. Use `precision_recall_curve` to find all candidate thresholds
4. Find highest precision threshold where spam recall = 1.0
5. Visualize threshold impact on metrics

**Output:**
- Optimal threshold value for Stage 1
- Precision-recall curve
- Confusion matrices at different thresholds
- Percentage of messages requiring Stage 2

### 3.2 Hybrid Cascade Implementation 🔄
**File:** `hybrid_cascade.py` (to be created)

**Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: SMS Message                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│ STAGE 1: TF-IDF Classifier (Fast, Free)                    │
│ - Vectorize with pre-trained TF-IDF                        │
│ - Get spam probability from best traditional ML model       │
│ - Decision:                                                 │
│   • p(spam) >= threshold_high  → SPAM (high confidence)    │
│   • p(spam) <= threshold_low   → HAM  (high confidence)    │
│   • Otherwise                  → UNCERTAIN (→ Stage 2)     │
└───────────────────────┬─────────────────────────────────────┘
                        │ (uncertain cases only, ~10-20%)
┌───────────────────────▼─────────────────────────────────────┐
│ STAGE 2: Embedding Classifier (Medium cost)                │
│ - Generate sentence transformer embedding                   │
│ - Classify with best embedding-based model                 │
│ - Decision:                                                 │
│   • p(spam) >= threshold_high  → SPAM                      │
│   • p(spam) <= threshold_low   → HAM                       │
│   • Otherwise                  → UNCERTAIN (→ Stage 3)     │
└───────────────────────┬─────────────────────────────────────┘
                        │ (very uncertain cases, ~1-5%)
┌───────────────────────▼─────────────────────────────────────┐
│ STAGE 3: LLM Classifier (Expensive, highest accuracy)      │
│ - Call Anthropic API with message                          │
│ - Parse binary yes/no response                             │
│ - Final classification                                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│ OUTPUT: Final Classification + Metadata                     │
│ - Label (spam/ham)                                          │
│ - Confidence scores from each stage                        │
│ - Stage that made final decision                           │
│ - Cost incurred                                             │
│ - Latency                                                   │
└─────────────────────────────────────────────────────────────┘
```

**Threshold Strategy:**
- **Stage 1 (TF-IDF):**
  - Low threshold: Optimized for 100% spam recall
  - High threshold: High precision for confident predictions
  - Uncertain zone: ~0.3-0.7 probability range

- **Stage 2 (Embeddings):**
  - Narrower uncertain zone (~0.4-0.6)
  - Higher confidence due to semantic understanding

- **Stage 3 (LLM):**
  - No uncertainty - binary decision
  - Handles edge cases and adversarial examples

**Implementation Details:**
```python
class HybridCascadeClassifier:
    def __init__(self, tfidf_model, embedding_model, llm_client):
        self.stage1 = tfidf_model
        self.stage2 = embedding_model
        self.stage3 = llm_client
        self.thresholds = self._load_optimal_thresholds()

    def predict(self, message):
        metadata = {
            'stages_used': [],
            'confidences': {},
            'cost': 0.0,
            'latency_ms': 0.0
        }

        # Stage 1: TF-IDF
        start = time.time()
        prob_stage1 = self.stage1.predict_proba(message)
        metadata['stages_used'].append(1)
        metadata['confidences']['stage1'] = prob_stage1
        metadata['latency_ms'] += (time.time() - start) * 1000

        if prob_stage1 >= self.thresholds['stage1_high']:
            return 'spam', metadata
        if prob_stage1 <= self.thresholds['stage1_low']:
            return 'ham', metadata

        # Stage 2: Embeddings (only if uncertain)
        start = time.time()
        prob_stage2 = self.stage2.predict_proba(message)
        metadata['stages_used'].append(2)
        metadata['confidences']['stage2'] = prob_stage2
        metadata['latency_ms'] += (time.time() - start) * 1000
        metadata['cost'] += EMBEDDING_COST  # Inference cost

        if prob_stage2 >= self.thresholds['stage2_high']:
            return 'spam', metadata
        if prob_stage2 <= self.thresholds['stage2_low']:
            return 'ham', metadata

        # Stage 3: LLM (only if still uncertain)
        start = time.time()
        result = self.stage3.classify(message)
        metadata['stages_used'].append(3)
        metadata['latency_ms'] += (time.time() - start) * 1000
        metadata['cost'] += self._calculate_llm_cost(message, result)

        return result, metadata
```

**Expected Performance:**
- Accuracy: Matches or exceeds best single approach
- Cost: 90-95% reduction vs. pure LLM (only 1-5% use LLM)
- Speed: Fast for majority (Stage 1), acceptable overall
- Recall: 100% for spam (by design)
- Precision: Improved by LLM on hard cases

---

## Phase 4: Comprehensive Benchmarking ⏳ PLANNED

### 4.1 Cost Analysis 🔄
**File:** `cost_benchmark.py` (to be created)

**Metrics to Compare:**
1. **Per-Message Cost:**
   - Traditional ML: ~$0 (local inference)
   - Embedding-based: ~$0 (local inference, one-time embedding cost)
   - LLM: ~$0.0001-0.001 per message
   - Hybrid: ~$0.00001-0.00005 per message (95% Stage 1, 4% Stage 2, 1% Stage 3)

2. **Infrastructure Costs:**
   - Model storage and loading
   - Memory requirements
   - GPU/CPU requirements

3. **Scaling Costs:**
   - Cost per 1K, 10K, 100K, 1M messages
   - Break-even analysis

### 4.2 Speed Benchmark 🔄
**File:** `speed_benchmark.py` (to be created)

**Metrics:**
- Single message latency (p50, p95, p99)
- Batch processing throughput
- Concurrent request handling
- Cold start vs. warm inference

### 4.3 Final Comparison 🔄
**File:** `final_comparison.py` (to be created)

**Comprehensive Report:**
```
┌─────────────┬──────────┬───────────┬──────────┬──────────┬────────────┐
│ Approach    │ Accuracy │ F1-Score  │ Cost/$   │ Latency  │ Use Case   │
├─────────────┼──────────┼───────────┼──────────┼──────────┼────────────┤
│ TF-IDF      │ 0.9750   │ 0.9550    │ $0       │ 1ms      │ Baseline   │
│ Embeddings  │ 0.9800   │ 0.9600    │ $0       │ 10ms     │ Semantic   │
│ LLM (Haiku) │ 0.9900   │ 0.9850    │ $0.50/1K │ 200ms    │ Accuracy   │
│ Hybrid      │ 0.9900   │ 0.9850    │ $0.02/1K │ 5ms avg  │ Production │
└─────────────┴──────────┴───────────┴──────────┴──────────┴────────────┘
```

**Visualizations:**
- Cost vs. Accuracy trade-off curve
- Latency distributions
- ROC curves for all approaches
- Precision-Recall curves
- Confusion matrices
- Stage usage distribution (for hybrid)

---

## Expected Deliverables

### Code Files
- [x] `embed.py` - Data preprocessing + embeddings
- [x] `traditional_ml.py` - TF-IDF baseline
- [x] `embedding_classifier.py` - Embedding baseline
- [x] `comparison.py` - Basic comparison
- [ ] `llm_classifier.py` - Direct LLM testing
- [ ] `threshold_optimizer.py` - Find optimal thresholds
- [ ] `hybrid_cascade.py` - Multi-stage classifier
- [ ] `cost_benchmark.py` - Cost analysis
- [ ] `speed_benchmark.py` - Latency analysis
- [ ] `final_comparison.py` - Comprehensive report

### Documentation
- [x] `README.md` - Project overview
- [x] `PROJECT_PLAN.md` - This file
- [x] `CLAUDE.md` - Claude Code guidance
- [ ] `HYBRID_ARCHITECTURE.md` - Detailed system design
- [ ] `COST_ANALYSIS.md` - Cost breakdown and projections

### Output Artifacts
- [ ] Complete performance metrics for all 4 approaches
- [ ] Cost projections for different scales
- [ ] Speed benchmarks
- [ ] Production deployment recommendations
- [ ] Academic paper/blog post draft

---

## Next Steps (Priority Order)

1. **Create `llm_classifier.py`**
   - Set up Anthropic API client
   - Implement spam classification prompts
   - Add cost and latency tracking
   - Run on full test set

2. **Create `threshold_optimizer.py`**
   - Load best TF-IDF model
   - Find optimal threshold for 100% spam recall
   - Visualize precision-recall trade-offs

3. **Create `hybrid_cascade.py`**
   - Implement 3-stage architecture
   - Add metadata tracking
   - Test on full dataset

4. **Benchmarking and Final Comparison**
   - Cost analysis across all approaches
   - Speed benchmarking
   - Generate comprehensive comparison report

5. **Documentation**
   - Write up findings
   - Create deployment guide
   - Publish results

---

## Success Criteria

### Performance
- [ ] Hybrid system achieves ≥99% accuracy
- [ ] 100% recall on spam messages
- [ ] ≥95% precision

### Cost
- [ ] Hybrid system costs <5% of pure LLM approach
- [ ] Cost per 1M messages < $50

### Speed
- [ ] Average latency < 10ms
- [ ] 95th percentile latency < 50ms

### Code Quality
- [ ] All scripts run end-to-end without errors
- [ ] Comprehensive logging and error handling
- [ ] Reproducible results with fixed random seeds
- [ ] Well-documented code and architecture
