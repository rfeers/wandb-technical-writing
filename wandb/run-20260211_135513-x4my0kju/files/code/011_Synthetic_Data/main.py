import weave
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import wandb


load_dotenv()
WEAVE_PROJECT = "synthetic-test-data"
weave.init(WEAVE_PROJECT)

wandb.init(
    entity="rfeers-databites",
    project=WEAVE_PROJECT,
)

client = OpenAI()

from typing import TypedDict, Literal

class TestCase(TypedDict):
    question: str
    answer: str
    difficulty: Literal["easy", "medium", "hard"]
    category: str
    metadata: dict

GENERATION_PROMPT = """Generate a customer support Q&A pair for an e-commerce platform.

Requirements:
- Question should be realistic, the kind a real customer would ask
- Answer must be accurate and helpful
- Vary the difficulty: easy (simple product questions), medium (account or order issues), hard (complex policy or technical problems)
- Cover different categories: products, shipping, returns, account, payment

Output valid JSON matching this schema:
{
  "question": "...",
  "answer": "...",
  "difficulty": "easy|medium|hard",
  "category": "products|shipping|returns|account|payment"
}

Make the examples diverse. Avoid repetitive phrasing."""

@weave.op()
def generate_test_case() -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": GENERATION_PROMPT}],
        temperature=0.8,
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
        data["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "model": "gpt-4o-mini"
        }
        return data
    except json.JSONDecodeError:
        return {"error": "Invalid JSON generated"}
    

@weave.op()
def generate_batch(n: int = 50) -> list[dict]:
    examples = []
    for i in range(n):
        example = generate_test_case()
        if "error" not in example:
            examples.append(example)
            print(f"Generated {i+1}/{n}")
    return examples

raw_examples = generate_batch(50)
print(f"\nGenerated {len(raw_examples)} examples")

JUDGE_PROMPT = """Evaluate this customer support Q&A pair for quality.

Question: {question}
Answer: {answer}
Difficulty: {difficulty}
Category: {category}

Check:
1. Is the question realistic? Would a real customer ask this?
2. Is the answer correct and helpful?
3. Does the difficulty rating make sense?
4. Does it match the stated category?
5. Is there enough specificity to be useful for testing?

Respond with valid JSON:
{{
  "pass": true/false,
  "score": 1-5,
  "issues": ["list", "of", "problems"],
  "reasoning": "brief explanation"
}}

Pass threshold: score >= 4, realistic question, correct answer, proper categorization."""

@weave.op()
def judge_quality(example: dict) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=example["question"],
        answer=example["answer"],
        difficulty=example["difficulty"],
        category=example["category"]
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

@weave.op()
def filter_batch(examples: list[dict], threshold: float = 4.0) -> tuple[list[dict], list[dict]]:
    passed = []
    failed = []
    
    for example in examples:
        judgment = judge_quality(example)
        example["quality_check"] = judgment
        
        if judgment.get("pass", False) and judgment.get("score", 0) >= threshold:
            passed.append(example)
        else:
            failed.append(example)
    
    return passed, failed

good_examples, bad_examples = filter_batch(raw_examples)
print(f"\nPassed: {len(good_examples)}")
print(f"Failed: {len(bad_examples)}")

@weave.op()
def remove_duplicates(examples: list[dict], similarity_threshold: float = 0.9) -> list[dict]:
    """Remove near-duplicate questions using simple token overlap."""
    unique = []
    seen_questions = []
    
    for ex in examples:
        question_tokens = set(ex["question"].lower().split())
        is_duplicate = False
        
        for seen in seen_questions:
            seen_tokens = set(seen.lower().split())
            overlap = len(question_tokens & seen_tokens)
            union = len(question_tokens | seen_tokens)
            
            if union > 0 and (overlap / union) > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(ex)
            seen_questions.append(ex["question"])
    
    return unique

final_examples = remove_duplicates(good_examples)
print(f"\nAfter deduplication: {len(final_examples)}")

from weave import Dataset

@weave.op()
def create_golden_set(examples: list[dict], name: str = "support_qa_v1") -> Dataset:
    for i, ex in enumerate(examples):
        ex["id"] = f"{name}_{i:03d}"
    
    dataset = Dataset(name=name, rows=examples)
    weave.publish(dataset)
    
    print(f"\nPublished dataset '{name}' with {len(examples)} examples")
    return dataset

golden_set = create_golden_set(final_examples)

from weave import Evaluation

@weave.op()
def dummy_model(question: str) -> str:
    """Replace this with your actual model."""
    return "This is a placeholder answer."

@weave.op()
def exact_match(answer: str, output: str) -> dict:
    return {"exact_match": answer.strip().lower() == output.strip().lower()}

evaluation = Evaluation(
    dataset=golden_set,
    scorers=[exact_match]
)

results = evaluation.evaluate(dummy_model)

TARGETED_PROMPT = """Generate a HARD difficulty customer support Q&A about RETURNS policy.

The question should involve:
- International returns
- Outside the normal return window
- Multiple items with different policies

This is a difficult edge case that requires deep policy knowledge.

Output valid JSON:
{{
  "question": "...",
  "answer": "...",
  "difficulty": "hard",
  "category": "returns"
}}"""

@weave.op()
def generate_targeted(prompt: str, n: int = 10) -> list[dict]:
    examples = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            response_format={"type": "json_object"}
        )
        try:
            data = json.loads(response.choices[0].message.content)
            data["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "model": "gpt-4o-mini",
                "targeted": True
            }
            examples.append(data)
        except:
            continue
    return examples

edge_cases = generate_targeted(TARGETED_PROMPT, 10)
filtered_edges, _ = filter_batch(edge_cases, threshold=4.0)

print(f"Generated {len(filtered_edges)} targeted edge cases")