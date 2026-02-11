import weave
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
import hashlib
import wandb

load_dotenv()
WEAVE_PROJECT = "synthetic-test-data"
weave.init(WEAVE_PROJECT)
wandb.init(
    entity="rfeers-databites",
    project=WEAVE_PROJECT,
)
client = OpenAI()

SCENARIOS = [
    {
        "name": "factual_lookup",
        "description": "Simple factual questions with clear answers",
        "instructions": """Generate question-answer pairs where:
- Question asks for a specific fact (date, person, place, number)
- Answer is 1-3 sentences, factually correct
- Context provided contains the answer clearly
- No ambiguity or interpretation needed""",
        "count": 15
    },
    {
        "name": "ambiguous_context",
        "description": "Questions where context is unclear or contradictory",
        "instructions": """Generate question-answer pairs where:
- Context contains contradictory information or is vague
- Correct answer acknowledges the ambiguity
- Answer cites which parts of context are unclear
- Tests whether system admits uncertainty""",
        "count": 10
    },
    {
        "name": "out_of_scope",
        "description": "Questions that cannot be answered from context",
        "instructions": """Generate question-answer pairs where:
- Question is reasonable but context doesn't contain the answer
- Correct response is 'I don't know' or similar refusal
- Context is related but missing key information
- Tests whether system hallucinates when it shouldn't""",
        "count": 10
    },
    {
        "name": "multi_hop_reasoning",
        "description": "Questions requiring multiple pieces of information",
        "instructions": """Generate question-answer pairs where:
- Answer requires combining 2-3 facts from context
- Context provides the facts but doesn't explicitly state the answer
- Answer shows reasoning steps
- Tests basic inference capability""",
        "count": 10
    }
]

@weave.op()
def generate_examples(scenario: Dict, temperature: float = 0.8) -> List[Dict]:
    """Generate synthetic examples for a given scenario."""
    
    prompt = f"""You are creating test data for evaluating a Q&A system.

Scenario: {scenario['name']}
Description: {scenario['description']}

{scenario['instructions']}

Generate {scenario['count']} distinct examples. For each example, provide:
- question: The user's question
- context: A short passage (2-4 sentences) that the system would use to answer
- reference_answer: The correct answer given the context

Return ONLY a JSON object with a single key "examples" containing an array of objects.
Each object must have 'question', 'context', and 'reference_answer' fields.

Make examples diverse. Vary topics, sentence structure, and difficulty. Avoid repetitive patterns."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    try:
        content = response.choices[0].message.content
        data = json.loads(content)
        examples = data.get("examples", [])
        
        # Add metadata
        for ex in examples:
            ex["scenario"] = scenario["name"]
            ex["generated_by"] = "gpt-4o-mini"
        
        return examples
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON for scenario {scenario['name']}: {e}")
        return []
    
all_examples = []

for scenario in SCENARIOS:
    print(f"Generating {scenario['count']} examples for {scenario['name']}...")
    examples = generate_examples(scenario)
    all_examples.extend(examples)
    print(f"  Generated {len(examples)} examples")

print(f"\nTotal generated: {len(all_examples)} examples")

@weave.op()
def filter_by_length(examples: List[Dict], 
                      min_question_len: int = 5,
                      min_context_len: int = 20,
                      min_answer_len: int = 3) -> List[Dict]:
    """Remove examples that are too short to be useful."""
    filtered = []
    for ex in examples:
        q_len = len(ex.get("question", "").split())
        c_len = len(ex.get("context", "").split())
        a_len = len(ex.get("reference_answer", "").split())
        
        if q_len >= min_question_len and c_len >= min_context_len and a_len >= min_answer_len:
            filtered.append(ex)
        else:
            print(f"  Filtered out example: q_len={q_len}, c_len={c_len}, a_len={a_len}")
    
    return filtered

@weave.op()
def filter_duplicates(examples: List[Dict]) -> List[Dict]:
    """Remove near-duplicate examples based on question similarity."""
    seen_hashes = set()
    filtered = []
    
    for ex in examples:
        # Simple hash of normalized question
        q_normalized = ex.get("question", "").lower().strip()
        q_hash = hashlib.md5(q_normalized.encode()).hexdigest()
        
        if q_hash not in seen_hashes:
            seen_hashes.add(q_hash)
            filtered.append(ex)
    
    return filtered

@weave.op()
def filter_format_errors(examples: List[Dict]) -> List[Dict]:
    """Remove examples missing required fields."""
    required_fields = ["question", "context", "reference_answer", "scenario"]
    filtered = []
    
    for ex in examples:
        if all(field in ex and ex[field] for field in required_fields):
            filtered.append(ex)
        else:
            missing = [f for f in required_fields if f not in ex or not ex[f]]
            print(f"  Filtered out example missing fields: {missing}")
    
    return filtered

print(f"\nBefore filtering: {len(all_examples)} examples")

filtered = filter_format_errors(all_examples)
print(f"After format check: {len(filtered)} examples ({len(all_examples) - len(filtered)} removed)")

filtered = filter_by_length(filtered)
print(f"After length check: {len(filtered)} examples")

filtered = filter_duplicates(filtered)
print(f"After deduplication: {len(filtered)} examples")

print(f"\nFinal count: {len(filtered)} examples")

# Exit early if no examples survived filtering
if len(filtered) == 0:
    print("\nERROR: All examples were filtered out. Check your generation prompt and filter thresholds.")
    exit(1)

VALIDATION_PROMPT = """You are evaluating synthetic test data for a Q&A system.

For each example, rate it on these criteria (1-5 scale each):

1. **Context Quality**: Does the context contain enough information? Is it coherent and realistic?
2. **Question Quality**: Is the question clear and natural? Would a real user ask this?
3. **Answer Correctness**: Given only the context, is the reference answer correct and appropriate?
4. **Overall Usefulness**: Would this example actually test what it's supposed to test?

Example to evaluate:
Question: {question}
Context: {context}
Reference Answer: {reference_answer}
Scenario: {scenario}

Return a JSON object with:
- context_quality (1-5)
- question_quality (1-5)  
- answer_correctness (1-5)
- overall_usefulness (1-5)
- reasoning (brief explanation)
- accept (true/false) - whether this example should be kept"""

@weave.op()
def validate_example(example: Dict) -> Dict:
    """Use an LLM judge to evaluate example quality."""
    
    prompt = VALIDATION_PROMPT.format(
        question=example.get("question", ""),
        context=example.get("context", ""),
        reference_answer=example.get("reference_answer", ""),
        scenario=example.get("scenario", "")
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    try:
        scores = json.loads(response.choices[0].message.content)
        return {
            "example_id": example.get("question", "")[:50],
            "scores": scores,
            "original_example": example
        }
    except json.JSONDecodeError:
        return {
            "example_id": example.get("question", "")[:50],
            "scores": {"accept": False, "reasoning": "Failed to parse validation"},
            "original_example": example
        }
    
print("\nRunning LLM-as-a-judge validation...")

validated = []
for i, ex in enumerate(filtered):
    if i % 10 == 0:
        print(f"  Validating example {i+1}/{len(filtered)}...")
    result = validate_example(ex)
    validated.append(result)

# Filter to accepted examples
accepted = [v["original_example"] for v in validated if v["scores"].get("accept", False)]

print(f"\nValidation results:")
print(f"  Reviewed: {len(validated)}")
print(f"  Accepted: {len(accepted)}")

# Avoid division by zero
if len(validated) > 0:
    rejection_rate = (1 - len(accepted)/len(validated))*100
    print(f"  Rejection rate: {rejection_rate:.1f}%")
else:
    print(f"  Rejection rate: N/A (no examples to validate)")

# Exit if no examples accepted
if len(accepted) == 0:
    print("\nERROR: All examples were rejected by the judge. Review your generation quality.")
    exit(1)

import random

def human_review_sample(examples: List[Dict], sample_size: int = 10):
    """Randomly sample examples for human review."""
    
    sample = random.sample(examples, min(sample_size, len(examples)))
    reviewed = []
    
    print(f"\n{'='*60}")
    print("HUMAN REVIEW")
    print(f"{'='*60}\n")
    
    for i, ex in enumerate(sample, 1):
        print(f"\nExample {i}/{len(sample)}")
        print(f"Scenario: {ex.get('scenario', 'unknown')}")
        print(f"\nQuestion: {ex['question']}")
        print(f"\nContext: {ex['context']}")
        print(f"\nReference Answer: {ex['reference_answer']}")
        
        while True:
            decision = input("\nKeep this example? (y/n/s to stop): ").strip().lower()
            if decision in ['y', 'n', 's']:
                break
        
        if decision == 's':
            break
        
        if decision == 'y':
            reviewed.append(ex)
    
    return reviewed

# Run human review on a sample
if len(accepted) > 0:
    print(f"\nYou have {len(accepted)} accepted examples.")
    do_review = input("Run human review on a sample? (y/n): ").strip().lower()
    
    if do_review == 'y':
        human_approved = human_review_sample(accepted, sample_size=10)
        print(f"\nYou approved {len(human_approved)}/{len(accepted)} reviewed examples")
        final_examples = human_approved if len(human_approved) > 0 else accepted
    else:
        final_examples = accepted
else:
    final_examples = accepted

from weave import Dataset

# Add unique IDs
for i, ex in enumerate(final_examples):
    ex["id"] = f"synthetic_{ex['scenario']}_{i:03d}"

# Create Weave dataset
dataset = Dataset(
    name="qa_synthetic_v1",
    rows=final_examples
)

weave.publish(dataset)

print(f"\n{'='*60}")
print(f"Published dataset: qa_synthetic_v1")
print(f"Total examples: {len(final_examples)}")
print(f"\nBreakdown by scenario:")

scenario_counts = {}
for ex in final_examples:
    scenario = ex["scenario"]
    scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1

for scenario, count in sorted(scenario_counts.items()):
    print(f"  {scenario}: {count}")

print("\n" + "="*60)
print("EXPANDING DATASET WITH NEW SCENARIO")
print("="*60 + "\n")

# After running an evaluation, you discover your model struggles 
# with questions that require calendar date reasoning

NEW_SCENARIO = {
    "name": "date_reasoning",
    "description": "Questions involving dates, durations, or calendar calculations",
    "instructions": """Generate question-answer pairs where:
- Question asks about a date, duration, or time relationship
- Context provides relevant dates
- Answer requires basic date arithmetic or comparison
- Tests whether system can handle temporal reasoning""",
    "count": 15
}

# Re-run generation just for this scenario
print(f"Generating {NEW_SCENARIO['count']} examples for {NEW_SCENARIO['name']}...")
new_examples = generate_examples(NEW_SCENARIO)
print(f"  Generated {len(new_examples)} examples")

new_filtered = filter_format_errors(new_examples)
new_filtered = filter_by_length(new_filtered)
new_filtered = filter_duplicates(new_filtered)

print(f"After filtering: {len(new_filtered)} examples")

if len(new_filtered) > 0:
    # Validate
    print("Validating new examples...")
    new_validated = []
    for ex in new_filtered:
        result = validate_example(ex)
        if result["scores"].get("accept", False):
            new_validated.append(result["original_example"])

    print(f"Accepted {len(new_validated)} new date reasoning examples")

    if len(new_validated) > 0:
        # Combine with existing dataset
        extended_examples = final_examples + new_validated

        # Publish as v2
        extended_dataset = Dataset(name="qa_synthetic_v2", rows=extended_examples)
        weave.publish(extended_dataset)
        
        print(f"\nPublished dataset: qa_synthetic_v2")
        print(f"Total examples: {len(extended_examples)}")
    else:
        print("\nNo new examples passed validation. Skipping v2 publication.")
else:
    print("\nNo new examples survived filtering. Skipping v2 publication.")