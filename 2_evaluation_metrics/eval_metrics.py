import os
import re
from dotenv import load_dotenv

import openai
from openai import OpenAI
from weave import Model
import weave

from weave import Dataset, Evaluation
from weave.scorers import EmbeddingSimilarityScorer, HallucinationFreeScorer
import sacrebleu
from rouge_score import rouge_scorer


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1) INITIALIZE OUR PROJECT
# Tip: use "team-name/project-name" if you work in a W&B team
weave.init("llm-metrics-demo")

# 2) EVALUATION DATASET - We create a dummy dataset
rows = [
    {
        "id": "1",
        "question": "What is the capital of France?",
        "context": "France's capital is Paris. It is known for the Seine and the Eiffel Tower.",
        "reference": "Paris",
    },
    {
        "id": "2",
        "question": "Who wrote Pride and Prejudice?",
        "context": "The novel Pride and Prejudice was authored by Jane Austen in 1813.",
        "reference": "Jane Austen",
    },
    {
        "id": "3",
        "question": "What year did Apollo 11 land on the Moon?",
        "context": "Apollo 11 landed on the Moon in 1969, with Armstrong and Aldrin walking on the surface.",
        "reference": "1969",
    },
]
# rows = [
#     {
#         "id": "1",
#         "question": "What is the capital of France?",
#         "context": "France's capital is Paris. It is known for the Seine and the Eiffel Tower.",
#         "reference": "The capital of France is Paris.",
#     },
#     {
#         "id": "2",
#         "question": "Who wrote Pride and Prejudice?",
#         "context": "The novel Pride and Prejudice was authored by Jane Austen in 1813.",
#         "reference": "Pride and Prejudice was written by Jane Austen.",
#     },
#     {
#         "id": "3",
#         "question": "What year did Apollo 11 land on the Moon?",
#         "context": "Apollo 11 landed on the Moon in 1969, with Armstrong and Aldrin walking on the surface.",
#         "reference": "Apollo 11 landed on the Moon in 1969.",
#     },
# ]


# We define a dataset with the dummy questions
dataset = Dataset(name="qa_metrics_demo_v1", rows=rows)
weave.publish(dataset)  # creates a versioned asset in the project

# 3) Wrap our model behind a simple interface
client = OpenAI()

class QAModel(Model):
    model_name: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a careful assistant. Use ONLY the provided context. "
        "If the answer is not in the context, reply exactly with: I don't know."
    )

    @weave.op()
    def predict(self, question: str, context: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the context below.\n"
                    f"Context:\n{context}\n\nQuestion: {question}"
                ),
            },
        ]
        res = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()

# 4) DEFINE EVALUATION METRICS
# 4a) Exact match (case/space-insensitive)
@weave.op()
def exact_match(reference: str, output: str) -> dict:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()
    return {"exact_match": norm(reference) == norm(output)}

# 4b) Simple token Jaccard overlap
@weave.op()
def jaccard(reference: str, output: str) -> dict:
    a, b = set(reference.lower().split()), set(output.lower().split())
    score = (len(a & b) / len(a | b)) if (a | b) else 0.0
    return {"jaccard": score}

# 4c) BLEU score (sentence-level, scaled to 0–1)
@weave.op()
def bleu(reference: str, output: str) -> dict:
    # sacrebleu expects: hypothesis string, list of reference strings
    score = sacrebleu.sentence_bleu(
        output,
        [reference],
        smooth_method="exp",
        tokenize="13a",   # default tokenizer; you can omit this line if you want
    ).score
    # sacrebleu returns 0–100; normalise to 0–1 for convenience
    return {"bleu": score / 100.0}

# 4d) ROUGE-L F1 (using rouge-score)
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

@weave.op()
def rouge_l(reference: str, output: str) -> dict:
    scores = _rouge.score(reference, output)
    # F-measure is the usual scalar people report
    return {"rougeL_f": scores["rougeL"].fmeasure}


# 4e) Embedding-based similarity to reference
sim_scorer = EmbeddingSimilarityScorer(
    model_id="openai/text-embedding-3-small",
    threshold=0.6,
    column_map={"target": "reference"},  # our dataset uses 'reference' as the label
)

# 4f) Hallucination check: does output introduce facts not in context?
hallucination_scorer = HallucinationFreeScorer(
    model_id="openai/gpt-4o",
    column_map={"context": "context"},  # dataset already uses 'context'
)

# 4g) LLM-as-judge
JUDGE_PROMPT = """You are grading an answer to a question.

Question:
{question}

Context (may or may not contain the answer):
{context}

Reference answer (may be short):
{reference}

Model answer:
{output}

Give a single score from 1 to 5:

1 = wrong or nonsensical
2 = partly correct but mostly wrong or missing
3 = roughly correct but incomplete or slightly unclear
4 = correct and reasonably clear
5 = fully correct, clear, and directly answers the question

Return ONLY the number."""

@weave.op()
def judge_quality(question: str, context: str, reference: str, output: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        reference=reference,
        output=output,
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",  # or a stronger judge if you prefer
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    raw = res.choices[0].message.content.strip()
    try:
        score = float(raw.split()[0])
    except ValueError:
        score = None
    return {"judge_quality_score": score}


# 5) Build an evaluation object
evaluation = Evaluation(
    dataset=dataset,
    scorers=[
        exact_match,
        jaccard,
        bleu,
        rouge_l,
        sim_scorer,
        hallucination_scorer,
        judge_quality
    ],
)


# 6) Run the evaluation and log metrics
async def run_eval():
    model = QAModel()
    await evaluation.evaluate(
        model,
        __weave={"display_name": "qa-metrics-demo-gpt4o-mini"},
    )

import asyncio
asyncio.run(run_eval())
