import weave
from weave import Dataset, Evaluation, Model
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
from openai import OpenAI
import os
import re
import numpy as np
from dotenv import load_dotenv

# STEP 1 -  Initialize the project
load_dotenv()

# Initialize Weave and capture the client for leaderboard creation
weave_client = weave.init("llm-eval-tutorial")
openai_client = OpenAI()

# STEP 2 - Create a diverse evaluation dataset

rows = [
    {
        "id": "1",
        "question": "What is the capital of Japan?",
        "reference": "Tokyo",
        "category": "factual"
    },
    {
        "id": "2",
        "question": "Who developed the theory of relativity?",
        "reference": "Albert Einstein",
        "category": "factual"
    },
    {
        "id": "3",
        "question": "Explain what photosynthesis is.",
        "reference": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar.",
        "category": "explanation"
    },
    {
        "id": "4",
        "question": "What is machine learning?",
        "reference": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.",
        "category": "explanation"
    },
    {
        "id": "5",
        "question": "Compare Python and JavaScript for web development.",
        "reference": "Python is primarily used for backend development with frameworks like Django and Flask, while JavaScript is essential for frontend development and can also handle backend with Node.js.",
        "category": "comparison"
    },
    {
        "id": "6",
        "question": "What's the difference between supervised and unsupervised learning?",
        "reference": "Supervised learning uses labeled training data to learn the mapping between inputs and outputs, while unsupervised learning finds patterns in unlabeled data without predefined categories.",
        "category": "comparison"
    },
    {
        "id": "7",
        "question": "Calculate 15% of 240.",
        "reference": "36",
        "category": "calculation"
    },
    {
        "id": "8",
        "question": "What is 7 multiplied by 13?",
        "reference": "91",
        "category": "calculation"
    },
    {
        "id": "9",
        "question": "What does HTTP stand for?",
        "reference": "Hypertext Transfer Protocol",
        "category": "factual"
    },
    {
        "id": "10",
        "question": "Describe the water cycle briefly.",
        "reference": "The water cycle involves evaporation of water from surfaces, condensation into clouds, precipitation as rain or snow, and collection back into bodies of water.",
        "category": "explanation"
    },
    {
        "id": "11",
        "question": "Compare RAM and ROM in computers.",
        "reference": "RAM is volatile memory used for temporary storage while programs run, whereas ROM is non-volatile memory that stores permanent instructions for the computer.",
        "category": "comparison"
    },
    {
        "id": "12",
        "question": "If a product costs $80 after a 20% discount, what was the original price?",
        "reference": "$100",
        "category": "calculation"
    }
]

dataset = Dataset(name="qa_benchmark_v2", rows=rows)
weave.publish(dataset)

# Step 4 - Define two model variants
class GPT4oMiniModel(Model):
    """GPT-4o-mini: More capable, better instruction following"""
    model_name: str = "gpt-4o-mini"

    @weave.op()
    def predict(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
            {"role": "user", "content": question}
        ]
        res = openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )
        return res.choices[0].message.content.strip()


class GPT35TurboModel(Model):
    """GPT-3.5-turbo: Faster, cheaper, less capable"""
    model_name: str = "gpt-3.5-turbo"

    @weave.op()
    def predict(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
            {"role": "user", "content": question}
        ]
        res = openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    
# STEP 4 - Implement evaluation metrics

# 4a) BLEU Score
import sacrebleu

@weave.op()
def bleu_score(reference: str, output: str) -> dict:
    """Statistical metric: measures n-gram overlap between output and reference."""
    score = sacrebleu.sentence_bleu(
        output,
        [reference],
        smooth_method="exp"
    ).score
    return {"bleu": score / 100.0}

# 4b) ROUGE-L
from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

@weave.op()
def rouge_l_score(reference: str, output: str) -> dict:
    """Statistical metric: measures longest common subsequence."""
    scores = _rouge.score(reference, output)
    return {"rouge_l": scores["rougeL"].fmeasure}

# 4c) BERTScore
from bert_score import score as bert_score_fn

@weave.op()
def bert_score(reference: str, output: str) -> dict:
    """Semantic metric: uses BERT embeddings to measure similarity."""
    try:
        _, _, F1 = bert_score_fn(
            [output],
            [reference],
            lang="en",
            verbose=False
        )
        return {"bert_score": float(F1[0])}
    except Exception as e:
        return {"bert_score": None}
    
# 4d) Embedding Similarity
@weave.op()
def embedding_similarity(reference: str, output: str) -> dict:
    """Semantic metric: computes cosine similarity between embeddings."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[reference, output]
    )
    ref_emb = np.array(response.data[0].embedding)
    out_emb = np.array(response.data[1].embedding)
    cosine_sim = np.dot(ref_emb, out_emb) / (np.linalg.norm(ref_emb) * np.linalg.norm(out_emb))
    return {"embedding_similarity": float(cosine_sim)}

#4e) LLM-as-a-Judge: Accuracy

ACCURACY_PROMPT = """You are evaluating the factual accuracy of an answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {output}

Rate the factual accuracy on a scale of 1-5:
1 = Completely incorrect or contradicts the reference
2 = Mostly incorrect with minor correct elements
3 = Partially correct but missing key information
4 = Mostly correct with minor issues
5 = Completely accurate and equivalent to the reference

Respond with ONLY a single number (1-5)."""

@weave.op()
def accuracy_judge(question: str, reference: str, output: str) -> dict:
    """LLM-as-a-judge metric focusing on factual correctness."""
    prompt = ACCURACY_PROMPT.format(question=question, reference=reference, output=output)
    res = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        normalized = min((score - 1) / 4, 0.9999)  # Normalize to 0-1
    except ValueError:
        normalized = None
    return {"accuracy_score": normalized}

# 4f) LLM-as-a-Judge: Helpfulness
HELPFULNESS_PROMPT = """You are evaluating how helpful an answer is to a user.

Question: {question}
Model Answer: {output}

Rate the helpfulness on a scale of 1-5:
1 = Not helpful at all, confusing or wrong
2 = Minimally helpful, lacks important context
3 = Somewhat helpful, adequate but could be clearer
4 = Helpful, clear and addresses the question well
5 = Extremely helpful, clear, complete, and well-explained

Respond with ONLY a single number (1-5)."""

@weave.op()
def helpfulness_judge(question: str, output: str) -> dict:
    """LLM-as-a-judge metric focusing on user helpfulness."""
    prompt = HELPFULNESS_PROMPT.format(question=question, output=output)
    res = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        normalized = (score - 1) / 4
    except ValueError:
        normalized = None
    return {"helpfulness_score": normalized}

# STEP 5 - Build the evaluation
evaluation = Evaluation(
    dataset=dataset,
    scorers=[
        bleu_score,
        rouge_l_score,
        bert_score,
        embedding_similarity,
        accuracy_judge,
        helpfulness_judge
    ]
)

# STEP 6 - RUN EVALUATION

import asyncio

async def run_all():
    print("Evaluating GPT-4o-mini...")
    await evaluation.evaluate(GPT4oMiniModel(), 
                              __weave={"display_name": "gpt-4o-mini"})
    
    print("Evaluating GPT-3.5-turbo...")
    await evaluation.evaluate(GPT35TurboModel(), 
                              __weave={"display_name": "gpt-3.5-turbo"})
    
    print("Creating leaderboard...")
    spec = leaderboard.Leaderboard(
        name="QA Model Comparison",
        description="Comparing GPT-4o-mini vs GPT-3.5-turbo across multiple metrics",
        columns=[
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="bleu_score",
                summary_metric_path="bleu.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="rouge_l_score",
                summary_metric_path="rouge_l.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="bert_score",
                summary_metric_path="bert_score.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="embedding_similarity",
                summary_metric_path="embedding_similarity.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="accuracy_judge",
                summary_metric_path="accuracy_score.mean",
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="helpfulness_judge",
                summary_metric_path="helpfulness_score.mean",
            ),
        ],
    )
    weave.publish(spec)
    print("Done! Check your W&B dashboard for results.")

asyncio.run(run_all())