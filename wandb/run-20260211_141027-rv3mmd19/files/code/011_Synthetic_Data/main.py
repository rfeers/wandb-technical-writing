import weave
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import wandb
from typing import TypedDict, Literal
import asyncio

load_dotenv()
WEAVE_PROJECT = "synthetic-test-data"
weave.init(WEAVE_PROJECT)

wandb.init(
    entity="rfeers-databites",
    project=WEAVE_PROJECT,
)

async_client = AsyncOpenAI()

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

Output valid JSON matching this EXACT schema:
{
  "question": "...",
  "answer": "...",
  "difficulty": "easy|medium|hard",
  "category": "products|shipping|returns|account|payment"
}

Make the examples diverse. Avoid repetitive phrasing.

IMPORTANT: Return ONLY the JSON object with these exact keys: question, answer, difficulty, category"""

def validate_example(data: dict) -> bool:
    """Check if generated example has required fields."""
    required_keys = ["question", "answer", "difficulty", "category"]
    
    if not isinstance(data, dict):
        return False
    
    for key in required_keys:
        if key not in data:
            print(f"Missing key: {key}")
            return False
        if not isinstance(data[key], str):
            print(f"Key {key} is not a string: {type(data[key])}")
            return False
        if not data[key].strip():
            print(f"Key {key} is empty")
            return False
    
    # Validate difficulty
    if data["difficulty"] not in ["easy", "medium", "hard"]:
        print(f"Invalid difficulty: {data['difficulty']}")
        return False
    
    # Validate category
    if data["category"] not in ["products", "shipping", "returns", "account", "payment"]:
        print(f"Invalid category: {data['category']}")
        return False
    
    return True

@weave.op()
async def generate_test_case() -> dict:
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": GENERATION_PROMPT}],
            temperature=0.8,
            response_format={"type": "json_object"},
            timeout=30.0
        )
        
        data = json.loads(response.choices[0].message.content)
        
        # Validate structure
        if not validate_example(data):
            print(f"Invalid example structure: {data}")
            return {"error": "Invalid structure"}
        
        data["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "model": "gpt-4o-mini"
        }
        return data
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return {"error": f"JSON decode error: {str(e)}"}
    except Exception as e:
        print(f"Error generating: {e}")
        return {"error": str(e)}

@weave.op()
async def generate_batch(n: int = 50) -> list[dict]:
    print(f"Generating {n} examples in parallel...")
    tasks = [generate_test_case() for _ in range(n)]
    results = await asyncio.gather(*tasks)
    
    examples = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    
    print(f"✅ Generated {len(examples)} valid examples")
    if errors:
        print(f"❌ {len(errors)} failed")
    
    return examples

async def main():
    raw_examples = await generate_batch(50)
    print(f"\nGenerated {len(raw_examples)} examples")
    
    if len(raw_examples) == 0:
        print("❌ No valid examples generated. Exiting.")
        return
    
    # Print first example to verify structure
    print("\nSample generated example:")
    print(json.dumps(raw_examples[0], indent=2))
    
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
    async def judge_quality(example: dict) -> dict:
        try:
            # Validate example has required keys before formatting
            required = ["question", "answer", "difficulty", "category"]
            for key in required:
                if key not in example:
                    print(f"Missing key in example for judging: {key}")
                    print(f"Example keys: {example.keys()}")
                    return {"pass": False, "score": 0, "issues": [f"Missing {key}"], "reasoning": "Invalid structure"}
            
            prompt = JUDGE_PROMPT.format(
                question=example["question"],
                answer=example["answer"],
                difficulty=example["difficulty"],
                category=example["category"]
            )
            
            response = await async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                timeout=30.0
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error judging example: {e}")
            print(f"Example that failed: {example}")
            return {"pass": False, "score": 0, "issues": [str(e)], "reasoning": "Error during evaluation"}
    
    @weave.op()
    async def filter_batch(examples: list[dict], threshold: float = 4.0) -> tuple[list[dict], list[dict]]:
        print("\nFiltering examples in parallel...")
        
        async def judge_and_classify(example: dict):
            judgment = await judge_quality(example)
            example["quality_check"] = judgment
            
            if judgment.get("pass", False) and judgment.get("score", 0) >= threshold:
                return ("passed", example)
            else:
                return ("failed", example)
        
        results = await asyncio.gather(*[judge_and_classify(ex) for ex in examples])
        
        passed = [ex for status, ex in results if status == "passed"]
        failed = [ex for status, ex in results if status == "failed"]
        
        return passed, failed
    
    good_examples, bad_examples = await filter_batch(raw_examples)
    print(f"\nPassed: {len(good_examples)}")
    print(f"Failed: {len(bad_examples)}")
    
    if len(good_examples) == 0:
        print("❌ No examples passed quality check. Exiting.")
        return
    
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
    
    print("\nRunning evaluation...")
    results = await evaluation.evaluate(dummy_model)
    print("Evaluation complete!")
    
    TARGETED_PROMPT = """Generate a HARD difficulty customer support Q&A about RETURNS policy.

The question should involve:
- International returns
- Outside the normal return window
- Multiple items with different policies

This is a difficult edge case that requires deep policy knowledge.

Output valid JSON matching this EXACT schema:
{{
  "question": "...",
  "answer": "...",
  "difficulty": "hard",
  "category": "returns"
}}

IMPORTANT: Return ONLY the JSON object with these exact keys: question, answer, difficulty, category"""
    
    @weave.op()
    async def generate_targeted(prompt: str, n: int = 10) -> list[dict]:
        print(f"\nGenerating {n} targeted examples...")
        
        async def gen_one():
            try:
                response = await async_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                    response_format={"type": "json_object"},
                    timeout=30.0
                )
                data = json.loads(response.choices[0].message.content)
                
                # Validate structure
                if not validate_example(data):
                    return {"error": "Invalid structure"}
                
                data["metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "model": "gpt-4o-mini",
                    "targeted": True
                }
                return data
            except Exception as e:
                print(f"Error: {e}")
                return {"error": str(e)}
        
        results = await asyncio.gather(*[gen_one() for _ in range(n)])
        valid = [r for r in results if "error" not in r]
        print(f"✅ Generated {len(valid)} valid targeted examples")
        return valid
    
    edge_cases = await generate_targeted(TARGETED_PROMPT, 10)
    if edge_cases:
        filtered_edges, _ = await filter_batch(edge_cases, threshold=4.0)
        print(f"\nGenerated {len(filtered_edges)} targeted edge cases")
    else:
        print("\n❌ No valid targeted examples generated")
    
    print("\n✅ Script complete!")

# Run everything
asyncio.run(main())