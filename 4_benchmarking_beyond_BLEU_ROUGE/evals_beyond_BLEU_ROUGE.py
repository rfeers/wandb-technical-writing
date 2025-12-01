# 1) STEP 1 - Initialize Weave
import weave
from openai import OpenAI
import os
from dotenv import load_dotenv
import asyncio
from weave import Dataset, Model, Evaluation

# Import for metrics
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
import numpy as np

load_dotenv()

# FIXED: Capture the Weave client
weave_client = weave.init("llm-benchmarking-demo")
openai_client = OpenAI()

#2) STEP 2 - Create Dataset
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

#3) STEP 3 - Define Two Models
class GPT4oMiniModel(Model):
    """GPT-4o-mini: More capable, better instruction following"""
    model_name: str = "gpt-4o-mini"
    
    @weave.op()
    def predict(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer questions accurately and concisely."},
            {"role": "user", "content": question}
        ]
        res = openai_client.chat.completions.create(  # FIXED: Use openai_client
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
        res = openai_client.chat.completions.create(  # FIXED: Use openai_client
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )
        return res.choices[0].message.content.strip()
    
#4) Step 4: Implement Metrics
#4a) BLEU Score
@weave.op()
def bleu_score(reference: str, output: str) -> dict:
    """
    Statistical metric: measures n-gram overlap between output and reference.
    Good for: detecting catastrophic failures, regression testing.
    Bad for: paraphrased but correct answers.
    """
    score = sacrebleu.sentence_bleu(
        output,
        [reference],
        smooth_method="exp"
    ).score
    return {"bleu": score / 100.0}

#4b) ROUGE-L
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

@weave.op()
def rouge_l_score(reference: str, output: str) -> dict:
    """
    Statistical metric: measures longest common subsequence.
    Good for: summarization tasks, keyword preservation.
    Bad for: restructured but semantically identical answers.
    """
    scores = _rouge.score(reference, output)
    return {"rouge_l": scores["rougeL"].fmeasure}

#4c) BERTScore
@weave.op()
def bert_score(reference: str, output: str) -> dict:
    """
    Semantic metric: uses BERT embeddings to measure similarity.
    Good for: recognizing paraphrases, semantic equivalence.
    Bad for: factual correctness (can score high on fluent nonsense).
    """
    try:
        _, _, F1 = bert_score_fn(
            [output],
            [reference],
            lang="en",
            verbose=False
        )
        return {"bert_score": float(F1[0])}
    except Exception as e:
        # If BERTScore fails, return None so evaluation continues
        print(f"BERTScore failed: {e}")
        return {"bert_score": None}

# 4d) Embedding Similarity
@weave.op()
def embedding_similarity(reference: str, output: str) -> dict:
    """
    Semantic metric: computes cosine similarity between embeddings.
    Good for: measuring semantic closeness in vector space.
    Bad for: doesn't check factual accuracy or logical correctness.
    """
    # Get embeddings for both texts
    response = openai_client.embeddings.create(  # FIXED: Use openai_client
        model="text-embedding-3-small",
        input=[reference, output]
    )
    
    ref_embedding = np.array(response.data[0].embedding)
    out_embedding = np.array(response.data[1].embedding)
    
    # Compute cosine similarity
    cosine_sim = np.dot(ref_embedding, out_embedding) / (
        np.linalg.norm(ref_embedding) * np.linalg.norm(out_embedding)
    )
    
    return {"embedding_similarity": float(cosine_sim)}

# 4e) LLM-as-a-Judge: Factual Accuracy (GPT-4o-mini)
ACCURACY_JUDGE_PROMPT = """You are evaluating the factual accuracy of an answer.

Question: {question}
Reference Answer: {reference}
Model Answer: {output}

Rate the factual accuracy on a scale of 1-5:
1 = Completely incorrect or contradicts the reference
2 = Mostly incorrect with minor correct elements
3 = Partially correct but missing key information
4 = Mostly correct with minor issues
5 = Completely accurate and equivalent to the reference

Consider:
- Are the core facts correct?
- Does it contradict the reference answer?
- Is critical information missing?

Respond with ONLY a single number (1-5)."""

@weave.op()
def accuracy_judge(question: str, reference: str, output: str) -> dict:
    """
    LLM-as-a-judge metric focusing on factual correctness.
    Uses GPT-4o-mini for cost-effective evaluation.
    Good for: detecting factual errors, hallucinations.
    Bad for: can be inconsistent across runs, inherits model biases.
    """
    prompt = ACCURACY_JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        output=output
    )
    res = openai_client.chat.completions.create(  # FIXED: Use openai_client
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        # Normalize to 0-1 range
        normalized_score = min((score - 1) / 4, 0.9999)
    except ValueError:
        normalized_score = None
    return {
        "accuracy_score": normalized_score,
        "accuracy_raw": score if normalized_score is not None else None
    }

#4f) LLM-as-a-Judge: Helpfulness (GPT-3.5-turbo)
HELPFULNESS_JUDGE_PROMPT = """You are evaluating how helpful an answer is to a user.

Question: {question}
Model Answer: {output}

Rate the helpfulness on a scale of 1-5:
1 = Not helpful at all, confusing or wrong
2 = Minimally helpful, lacks important context
3 = Somewhat helpful, adequate but could be clearer
4 = Helpful, clear and addresses the question well
5 = Extremely helpful, clear, complete, and well-explained

Consider:
- Does it directly address the question?
- Is it clear and easy to understand?
- Does it provide enough context/explanation?
- Would this satisfy a user asking this question?

Respond with ONLY a single number (1-5)."""

@weave.op()
def helpfulness_judge(question: str, output: str) -> dict:
    """
    LLM-as-a-judge metric focusing on user helpfulness.
    Uses GPT-3.5-turbo for different perspective and lower cost.
    Good for: evaluating user experience, clarity, completeness.
    Bad for: subjective, may not catch subtle factual errors.
    """
    prompt = HELPFULNESS_JUDGE_PROMPT.format(
        question=question,
        output=output
    )
    res = openai_client.chat.completions.create(  # FIXED: Use openai_client
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        score = float(res.choices[0].message.content.strip())
        # Normalize to 0-1 range
        normalized_score = (score - 1) / 4
    except ValueError:
        normalized_score = None
    return {
        "helpfulness_score": normalized_score,
        "helpfulness_raw": score if normalized_score is not None else None
    }

#Step 5: Build the Evaluation
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

#Step 6: Run Evaluations and Create Leaderboard
async def run_all():
    # First, run at least one evaluation to create the evaluation object in Weave
    print("Running evaluations...")
    
    # Initialize models
    gpt4o_mini_model = GPT4oMiniModel()
    gpt35_turbo_model = GPT35TurboModel()
    
    # Run evaluations - this will create the evaluation object in Weave
    print("\nEvaluating GPT-4o-mini...")
    result1 = await evaluation.evaluate(gpt4o_mini_model)
    
    print("\nEvaluating GPT-3.5-turbo...")
    result2 = await evaluation.evaluate(gpt35_turbo_model)
    
    print("\n Evaluations completed!")
    
    # NOW create and publish the leaderboard AFTER evaluations are done
    print("\nCreating leaderboard...")
    
    # Import leaderboard modules
    from weave.flow import leaderboard
    from weave.trace.ref_util import get_ref
    
    # Create the leaderboard specification
    leaderboard_spec = leaderboard.Leaderboard(
        name="LLM Benchmarking Leaderboard",
        description="Comparing GPT-4o-mini vs GPT-3.5-turbo on Q&A tasks across multiple metrics",
        columns=[
            # BLEU Score
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="bleu_score",
                summary_metric_path="bleu.mean",
            ),
            # ROUGE-L Score
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="rouge_l_score",
                summary_metric_path="rouge_l.mean",
            ),
            # BERTScore
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="bert_score",
                summary_metric_path="bert_score.mean",
            ),
            # Embedding Similarity
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="embedding_similarity",
                summary_metric_path="embedding_similarity.mean",
            ),
            # Accuracy (LLM Judge)
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="accuracy_judge",
                summary_metric_path="accuracy_score.mean",
            ),
            # Helpfulness (LLM Judge)
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="helpfulness_judge",
                summary_metric_path="helpfulness_score.mean",
            ),
        ]
    )
    
    # Publish the leaderboard
    published_leaderboard = weave.publish(leaderboard_spec)
    print(f" Leaderboard published: {published_leaderboard}")
    
    # Retrieve and print results - FIXED: Use weave_client
    results = leaderboard.get_leaderboard_results(leaderboard_spec, weave_client)
    print("\n=== Leaderboard Results ===")
    print(results)

asyncio.run(run_all())