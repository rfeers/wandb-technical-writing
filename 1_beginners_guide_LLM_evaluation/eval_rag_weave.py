import os, asyncio, re
import weave
from weave import Model, Dataset, Evaluation
from openai import OpenAI
from dotenv import load_dotenv
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

# Load .env file (by default it looks in the current working directory)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # or os.environ["API_KEY"] if you want it to error when missing

# ---- 0) Project init ---------------------------------------------------------
# Tip: use "team-name/project-name" to log under a W&B team
weave.init("llm-eval-demo")

# ---- 1) Tiny evaluation dataset ---------------------------------------------
# Each row contains: a question, a short context that actually contains the answer,
# and a reference answer we consider "correct".
rows = [
    {
        "id": "1",
        "question": "What is the capital of France?",
        "context": "France's capital is Paris. It is known for the Seine and the Eiffel Tower.",
        "reference": "Paris"
    },
    {
        "id": "2",
        "question": "Who wrote Pride and Prejudice?",
        "context": "The novel Pride and Prejudice was authored by Jane Austen in 1813.",
        "reference": "Jane Austen"
    },
    {
        "id": "3",
        "question": "What year did Apollo 11 land on the Moon?",
        "context": "Apollo 11 landed on the Moon in 1969, with Armstrong and Aldrin walking on the surface.",
        "reference": "1969"
    },
    {
        "id": "4",
        "question": "Which element has the chemical symbol 'Na'?",
        "context": "In the periodic table, 'Na' denotes sodium, which forms NaCl with chlorine.",
        "reference": "Sodium"
    },
    {
        "id": "5",
        "question": "Name the largest planet in our solar system.",
        "context": "Jupiter is the largest planet in our solar system.",
        "reference": "Jupiter"
    }
]

# Option A: Use a versioned Weave Dataset object (nice for UI & versioning)
dataset = Dataset(name="qa_eval_v1", rows=rows)
weave.publish(dataset)  # makes it appear/version in the Weave UI

# ---- 2) Two model variants to compare ---------------------------------------
client = OpenAI()

# Variant A: permissive prompt (more likely to hallucinate if context is weak)
class QAModelLoose(Model):
    model_name: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful assistant. Answer the user question clearly."
    )
    @weave.op()
    def predict(self, question: str, context: str) -> str:
        msg = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Question: {question}\nContext:\n{context}"},
        ]
        res = client.chat.completions.create(model=self.model_name, messages=msg, temperature=0.2)
        return res.choices[0].message.content.strip()

# Variant B: grounded prompt (only answer from the given context)
class QAModelGrounded(Model):
    model_name: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a careful assistant. Use ONLY the provided context. "
        "If the answer is not in context, reply exactly with: I don't know."
    )
    @weave.op()
    def predict(self, question: str, context: str) -> str:
        msg = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                "Answer the question using ONLY the context below.\n"
                "Context:\n" + context + "\n\nQuestion: " + question
            )},
        ]
        res = client.chat.completions.create(model=self.model_name, messages=msg, temperature=0.0)
        return res.choices[0].message.content.strip()

# ---- 3) Define evaluation metrics (Scorers) ---------------------------------
# 3a) Heuristic: Exact match (case/space-insensitive)
@weave.op()
def exact_match(reference: str, output: str) -> dict:
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s).strip().lower()
    return {"exact_match": norm(reference) == norm(output)}

# 3b) Heuristic: Jaccard token overlap
@weave.op()
def jaccard(reference: str, output: str) -> dict:
    a, b = set(reference.lower().split()), set(output.lower().split())
    return {"jaccard": (len(a & b) / len(a | b)) if (a | b) else 0.0}

# 3c) Built‑in: Embedding similarity (semantic match to reference)
from weave.scorers import EmbeddingSimilarityScorer
sim_scorer = EmbeddingSimilarityScorer(
    model_id="openai/text-embedding-3-small",  # any LiteLLM-supported embedding model
    threshold=0.7,
    column_map={"target": "reference"}  # our dataset uses 'reference' instead of 'target'
)

# 3d) Built‑in: Hallucination‑free check (judge if answer invents facts not in context)
from weave.scorers import HallucinationFreeScorer
hallucination_scorer = HallucinationFreeScorer(
    model_id="openai/gpt-4o",
    # This scorer expects 'context' and 'output'; we map if names differ.
    column_map={"context": "context"}  # (output is auto-mapped)
)

# 3e) (Optional) Built‑in moderation
# from weave.scorers import OpenAIModerationScorer
# moderation_scorer = OpenAIModerationScorer()

# ---- 4) Build the Evaluation -------------------------------------------------
evaluation = Evaluation(
    dataset=dataset,    # could also be `rows`
    scorers=[exact_match, jaccard, sim_scorer, hallucination_scorer]
)

# ---- 5) Run evaluations for both variants -----------------------------------
async def run_all():
    # You can optionally give runs a friendly display name for the UI
    await evaluation.evaluate(QAModelLoose(), __weave={"display_name": "loose_prompt"})
    await evaluation.evaluate(QAModelGrounded(), __weave={"display_name": "grounded_prompt"})

    spec = leaderboard.Leaderboard(
        name="qa_eval_leaderboard",
        description="Compare 'loose' vs 'grounded' prompts on QA metrics",
        columns=[
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="EmbeddingSimilarityScorer",  # name of the scorer
                summary_metric_path="similarity_score.mean",  # choose any summary field
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="HallucinationFreeScorer",
                summary_metric_path="has_hallucination.true_fraction",  # lower is better
            ),
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="exact_match",
                summary_metric_path="exact_match.true_fraction",
            ),
        ],
    )
    weave.publish(spec)

if __name__ == "__main__":
    asyncio.run(run_all())
    print("Done. Open the Weave project link printed in your console to explore.")
    
