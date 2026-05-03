<div align="center">

# MultiAgent-X
### Multilingual Agentic Function-Calling Benchmark for Under-Resourced Languages

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-orange.svg)](https://huggingface.co/datasets/Saurabh-66/MultiAgent-X)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)](https://www.kaggle.com/datasets/saurabhmallik/multiagent-x-multilingual-agentic-function-call)
[![Records](https://img.shields.io/badge/Records-10%2C551-red.svg)]()
[![Languages](https://img.shields.io/badge/Languages-12-green.svg)]()
[![Scripts](https://img.shields.io/badge/Scripts-7-orange.svg)]()
[![Speakers](https://img.shields.io/badge/Speakers-1.3B%2B-purple.svg)]()

**Created with [Adaptive Data by Adaption](https://www.adaptionlabs.ai/)**

*Submission for [The Uncharted Data Challenge by Adaption](https://www.kaggle.com/competitions/the-uncharted-data-challenge)*

</div>

---

![Geographic Coverage](/images/map.png)

---

## Overview

MultiAgent-X is the **first open-source multilingual function-calling training and evaluation dataset** targeting under-resourced languages. It contains 10,551 records across 12 languages, 7 unique writing systems, and 5 life-critical agentic domains covering a combined speaker population of over 1.3 billion people.

### The Problem

MASSIVE-Agents (EMNLP 2025) evaluated multilingual function-calling across 52 languages and 21 models. The top-performing model achieved **57.37% accuracy on English** but only **6.81% on Amharic**. For several languages, top models scored zero.

![Agentic Gap](/images/agentic_gap.png)

The cause is data. No open multilingual function-calling training dataset existed for these languages. BFCL is English only. MASSIVE-Agents released no training data. MultiAgent-X fills that gap.

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total records | 10,551 |
| Languages | 12 |
| Unique scripts | 7 |
| Agentic domains | 5 |
| Functions | 24 |
| Train / Dev / Test | 7,393 / 1,545 / 1,613 |
| Quality pass rate | 100% |

---

## Languages

![Language Distribution](/images/language_distribution.png)

| Language | Code | Script | Region | Speakers | Resource Level | Records |
|----------|------|--------|--------|----------|----------------|---------|
| Hindi | hi | Devanagari | Rural India | 600M | Medium | 984 |
| Punjabi | pa | Gurmukhi | India/Pakistan | 120M | Low | 984 |
| Khmer | km | Khmer | Cambodia | 16M | Very Low | 984 |
| Swahili | sw | Latin | East Africa | 200M | Medium | 980 |
| Sinhala | si | Sinhala | Sri Lanka | 17M | Low | 978 |
| Tigrinya | ti | Ethiopic | Ethiopia/Eritrea | 7M | Very Low | 975 |
| Igbo | ig | Latin | SE Nigeria | 27M | Low | 974 |
| Yoruba | yo | Latin | SW Nigeria | 47M | Low | 971 |
| Hausa | ha | Latin | Nigeria/Niger | 70M | Low | 968 |
| Amharic | am | Ethiopic | Ethiopia | 57M | Very Low | 919 |
| Lao | lo | Lao | Laos | 7M | Very Low | 454 |
| Oromo | om | Latin | Ethiopia/Kenya | 37M | Very Low | 380 |

![Speaker Coverage](/images/speaker_coverage.png)

---

## Domains and Functions

![Domain Distribution](/images/domain_type_distribution.png)

| Domain | Records | Functions |
|--------|---------|-----------|
| Healthcare | 2,388 | `book_clinic_appointment`, `check_medication_availability`, request_ambulance, get_vaccination_schedule, report_disease_outbreak |
| Agriculture | 2,347 | get_crop_disease_diagnosis, get_market_price, request_extension_worker, get_weather_planting_advice, register_cooperative_sale |
| Mobile Finance | 2,040 | send_mobile_money, check_balance, pay_bill, request_mobile_loan, get_exchange_rate |
| Emergency Response | 1,987 | report_emergency, request_evacuation, locate_nearest_shelter, request_food_distribution |
| Government Services | 1,789 | register_birth, check_document_status, report_infrastructure_issue, apply_for_social_support |

---

## Example Types

| Type | Count | Description |
|------|-------|-------------|
| positive_easy | 3,276 | Clear unambiguous requests |
| positive_medium | 1,958 | Requires inference and regional knowledge |
| positive_hard | 1,075 | Idiomatic and culturally indirect phrasing |
| multi_turn_q | 1,074 | Agent must request clarification before acting |
| multi_turn_a | 1,076 | User provides missing info; agent completes call |
| negative | 1,047 | No applicable function -- irrelevance detection |
| parallel | 530 | Two simultaneous function calls required |
| ambiguous | 515 | Required argument cannot be determined |

---

## Function Coverage

![Function Heatmap](/images/function_heatmap.png)

---

## Record Schema

```json
{
  "id": "max_healthcare_am_positive_easy_000016",
  "language_code": "am",
  "language_name": "Amharic",
  "script": "Ethiopic",
  "resource_level": "very_low",
  "domain": "healthcare",
  "example_type": "positive_easy",
  "difficulty": "easy",
  "ambiguity_type": "none",
  "user_utterance": "(natural language in target script)",
  "english_reference_utterance": "(English source utterance)",
  "cultural_context_note": "(what was adapted and why)",
  "expected_output": {
    "function_name": "book_clinic_appointment",
    "arguments": {
      "facility_type": "government_hospital",
      "urgency": "emergency",
      "district": "Kirkos",
      "condition_category": "general",
      "preferred_date": "today"
    }
  },
  "reasoning_trace": "(why this function was called)",
  "split": "train",
  "source": "synthetic_culturally_grounded",
  "created_with": "Adaptive Data by Adaption",
  "validated": false
}
```

---

## Quick Start

### Load the dataset

```python
from datasets import load_dataset

ds = load_dataset("Saurabh-66/MultiAgent-X")

train = ds["train"]
dev   = ds["validation"]
test  = ds["test"]

print(f"Train: {len(train)} | Dev: {len(dev)} | Test: {len(test)}")
```

### Filter by language

```python
amharic = ds["train"].filter(lambda x: x["language_code"] == "am")
print(f"Amharic training records: {len(amharic)}")
```

### Filter by domain

```python
healthcare = ds["train"].filter(lambda x: x["domain"] == "healthcare")
```

### Format for fine-tuning

```python
import json

SYSTEM_PROMPT = """You are an AI assistant that helps users by calling the appropriate function.
Given the user request and available functions, output the function call in JSON format.
If no function applies, output null.
If clarification is needed, output: {"function_name": "ask_clarification", "arguments": {"what_to_ask": "..."}}"""

def format_for_sft(record):
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": record["user_utterance"]},
            {"role": "assistant", "content": json.dumps(
                record["expected_output"], ensure_ascii=False
            )}
        ],
        "language": record["language_code"],
        "domain":   record["domain"]
    }

formatted = [format_for_sft(r) for r in ds["train"]]
```

### Evaluate predictions

```python
def ast_match(predicted, expected):
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False
    if isinstance(expected, list):
        if not isinstance(predicted, list) or len(predicted) != len(expected):
            return False
        return all(ast_match(p, e) for p, e in zip(predicted, expected))
    if predicted.get("function_name") != expected.get("function_name"):
        return False
    for key, val in expected.get("arguments", {}).items():
        if key not in predicted.get("arguments", {}):
            return False
        if str(predicted["arguments"][key]).lower() != str(val).lower():
            return False
    return True

# Example evaluation
correct = sum(ast_match(pred, record["expected_output"])
              for pred, record in zip(predictions, test_records))
accuracy = correct / len(test_records) * 100
print(f"AST Accuracy: {accuracy:.2f}%")
```

---

## Construction Pipeline

### Pipeline Overview

```
  English Seeds (984 records)
              |
              v
+----------------------------+
|      Qwen3-32B-AWQ         |
|  Cultural Seed Generation  |
|    5 domains x 8 types     |
+----------------------------+
              |
              v
+----------------------------+
|  Per-Language Adaptation   |
|       12 languages         |
|     Cultural blueprints    |
| Adaptation not translation |
+----------------------------+
              |
              v
+----------------------------+
|      Quality Filtering     |
|      Schema validation     |
|     Script verification    |
|        100% pass rate      |
+----------------------------+
              |
              v
+----------------------------+
|    MultiAgent-X Dataset    |
|       10,551 records       |
|     Train / Dev / Test     |
+----------------------------+
```

### Step 1: Seed Generation

984 English seeds generated using Qwen3-32B-AWQ covering all 5 domains and 8 example types, grounded in Sub-Saharan African and South Asian cultural contexts from the start.

### Step 2: Cultural Adaptation via Adaptive Data by Adaption

Each seed culturally adapted into 12 target languages with explicit per-language blueprints specifying:
- Regional services (M-Pesa vs Telebirr vs bKash vs OPay vs UPI)
- Local place names (Addis Ababa, Kano, Dhaka, Phnom Penh, Kathmandu)
- Local currencies (ETB, KES, BDT, NGN, NPR, KHR)
- Natural colloquial speech patterns

Adaptation, not translation. This is the principle Adaptive Data by Adaption is built around.

### Step 3: Quality Filtering

Rule-based validation on every record:
- JSON schema compliance
- Function name validity against the 24-function ontology
- Required argument presence
- Native script verification for non-Latin writing systems

### Step 4: Dataset Assembly

Hash-based deterministic 70/15/15 train/dev/test split for full reproducibility.

---

## Files

| File | Records | Description |
|------|---------|-------------|
| `train.jsonl` | 7,393 | Training split |
| `dev.jsonl` | 1,545 | Validation split |
| `test.jsonl` | 1,613 | Test split with ground truth |
| `stats.json` | - | Full dataset statistics |
| `schemas/functions.json` | - | Complete 24-function schema |
| `images/` | - | Visualizations and charts |

---

## Comparison with Related Work

| Dataset | Languages | Training Data | Agentic | Cultural Grounding |
|---------|-----------|--------------|---------|-------------------|
| BFCL | English only | Yes | Yes | No |
| MASSIVE-Agents | 52 | **No** | Yes | No |
| xLAM | English-dominant | Yes | Yes | No |
| **MultiAgent-X** | **12 (expanding)** | **Yes** | **Yes** | **Yes** |

---

## Roadmap

- [ ] Expand to 30+ languages across 6 continents
- [ ] Add education domain (school enrollment, exam access, scholarship)
- [ ] Add legal aid domain (rights access, domestic violence reporting)
- [ ] Add WASH domain (water, sanitation, hygiene)
- [ ] Human validation by native speaker annotators
- [ ] Fine-tuning experiments and baseline model release

---

## Citation

```bibtex
@dataset{multiagentx2026,
  title     = {MultiAgent-X: Multilingual Agentic Function-Calling Benchmark
               for Under-Resourced Languages},
  author    = {Mallik, Saurabh},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/Saurabh-66/MultiAgent-X},
  note      = {Created with Adaptive Data by Adaption.
               Submission for The Uncharted Data Challenge 2026.}
}
```

---

## Links

- Kaggle Writeup: https://www.kaggle.com/competitions/the-uncharted-data-challenge/writeups/multiagent-x
- HuggingFace Dataset: https://huggingface.co/datasets/Saurabh-66/MultiAgent-X
- Kaggle Dataset: https://www.kaggle.com/datasets/saurabhmallik/multiagent-x-multilingual-agentic-function-call
- Kaggle Competition: https://www.kaggle.com/competitions/the-uncharted-data-challenge
- Adaptive Data by Adaption: https://www.adaptionlabs.ai/

---

<div align="center">

*Created with [Adaptive Data by Adaption](https://www.adaptionlabs.ai/)*

*The Uncharted Data Challenge 2026*

</div>
