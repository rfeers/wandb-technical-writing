import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import weave
from weave import Dataset, Evaluation, Model

load_dotenv()
client = OpenAI()

weave.init("rag-evaluation-demo")

KNOWLEDGE_BASE = {
    "chunk_1": {
        "id": "chunk_1",
        "content": "Acme Corp offers a 30-day return policy for all products purchased online. "
                   "Items must be in original packaging and unused condition. Refunds are processed "
                   "within 5-7 business days after we receive the returned item.",
        "source": "return_policy.md"
    },
    "chunk_2": {
        "id": "chunk_2",
        "content": "Employee vacation policy: Full-time employees receive 15 days of paid vacation "
                   "per year, accruing at 1.25 days per month. Unused vacation days can be carried "
                   "over up to a maximum of 5 days into the next calendar year.",
        "source": "hr_handbook.md"
    },
    "chunk_3": {
        "id": "chunk_3",
        "content": "Technical support is available Monday through Friday, 9 AM to 6 PM Eastern Time. "
                   "For urgent issues outside business hours, premium support customers can call our "
                   "24/7 emergency hotline at 1-800-555-0199.",
        "source": "support_guide.md"
    },
    "chunk_4": {
        "id": "chunk_4",
        "content": "Shipping rates: Standard shipping (5-7 business days) is free for orders over $50. "
                   "Express shipping (2-3 business days) costs $12.99. Overnight shipping is available "
                   "for $24.99. International shipping rates vary by destination.",
        "source": "shipping_info.md"
    },
    "chunk_5": {
        "id": "chunk_5",
        "content": "Password requirements: All passwords must be at least 12 characters long and include "
                   "at least one uppercase letter, one lowercase letter, one number, and one special "
                   "character. Passwords expire every 90 days and cannot be reused within 12 months.",
        "source": "security_policy.md"
    }
}

rows = [
    {
        "id": "q1",
        "question": "What is the return policy for online purchases?",
        "ground_truth_context": ["chunk_1"],
        "reference": "Acme Corp offers a 30-day return policy for online purchases. Items must be in original packaging and unused. Refunds are processed within 5-7 business days."
    },
    {
        "id": "q2",
        "question": "How many vacation days do full-time employees get?",
        "ground_truth_context": ["chunk_2"],
        "reference": "Full-time employees receive 15 days of paid vacation per year, accruing at 1.25 days per month."
    },
    {
        "id": "q3",
        "question": "What are the technical support hours?",
        "ground_truth_context": ["chunk_3"],
        "reference": "Technical support is available Monday through Friday, 9 AM to 6 PM Eastern Time."
    },
    {
        "id": "q4",
        "question": "How much does express shipping cost?",
        "ground_truth_context": ["chunk_4"],
        "reference": "Express shipping costs $12.99 and takes 2-3 business days."
    },
    {
        "id": "q5",
        "question": "What are the password requirements?",
        "ground_truth_context": ["chunk_5"],
        "reference": "Passwords must be at least 12 characters with uppercase, lowercase, number, and special character. They expire every 90 days."
    },
    {
        "id": "q6",
        "question": "Can I get free shipping?",
        "ground_truth_context": ["chunk_4"],
        "reference": "Yes, standard shipping is free for orders over $50."
    },
    {
        "id": "q7",
        "question": "How long do refunds take to process?",
        "ground_truth_context": ["chunk_1"],
        "reference": "Refunds are processed within 5-7 business days after the returned item is received."
    },
    {
        "id": "q8",
        "question": "Can I carry over unused vacation days?",
        "ground_truth_context": ["chunk_2"],
        "reference": "Yes, up to 5 unused vacation days can be carried over into the next calendar year."
    }
]

dataset = Dataset(name="rag_eval_dataset_v1", rows=rows)
weave.publish(dataset)

@weave.op()
def get_embedding(text: str) -> list:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


@weave.op()
def retrieve(question: str, top_k: int = 2) -> list:
    question_embedding = get_embedding(question)
    
    similarities = []
    for chunk_id, chunk in KNOWLEDGE_BASE.items():
        chunk_embedding = get_embedding(chunk["content"])
        similarity = np.dot(question_embedding, chunk_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((chunk_id, similarity, chunk["content"]))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [
        {"chunk_id": s[0], "content": s[2], "score": float(s[1])}
        for s in similarities[:top_k]
    ]


class RAGSystem(Model):
    model_name: str = "gpt-4o-mini"
    top_k: int = 2
    
    @weave.op()
    def predict(self, question: str) -> dict:
        retrieved_chunks = retrieve(question, self.top_k)
        
        context = "\n\n".join([
            f"[Source: {chunk['chunk_id']}]\n{chunk['content']}"
            for chunk in retrieved_chunks
        ])
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions based only on the "
                           "provided context. If the context doesn't contain the answer, say so."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "retrieved_chunks": [c["chunk_id"] for c in retrieved_chunks],
            "context": context
        }

@weave.op()
def context_precision(ground_truth_context: list, model_output: dict) -> dict:
    retrieved = set(model_output.get("retrieved_chunks", []))
    relevant = set(ground_truth_context)
    
    if not retrieved:
        return {"context_precision": 0.0}
    
    relevant_retrieved = retrieved.intersection(relevant)
    precision = len(relevant_retrieved) / len(retrieved)
    
    return {"context_precision": precision}


@weave.op()
def context_recall(ground_truth_context: list, model_output: dict) -> dict:
    retrieved = set(model_output.get("retrieved_chunks", []))
    relevant = set(ground_truth_context)
    
    if not relevant:
        return {"context_recall": 1.0}
    
    relevant_retrieved = retrieved.intersection(relevant)
    recall = len(relevant_retrieved) / len(relevant)
    
    return {"context_recall": recall}

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Context:
{context}

Question: {question}

Answer: {answer}

Rate faithfulness from 1-5:
1 = Contains significant information not in context (hallucination)
2 = Contains some unsupported claims
3 = Mostly supported but has minor unsupported details
4 = Well-supported with only trivial additions
5 = Completely faithful to the context

Return only a number."""


RELEVANCY_PROMPT = """You are evaluating whether an answer is relevant to the question.

Question: {question}

Answer: {answer}

Reference: {reference}

Rate relevancy from 1-5:
1 = Does not address the question
2 = Partially addresses but misses key points
3 = Addresses question but incomplete or has irrelevant parts
4 = Addresses well with minor issues
5 = Perfectly addresses the question

Return only a number."""


@weave.op()
def faithfulness_score(question: str, model_output: dict) -> dict:
    answer = model_output.get("answer", "")
    context = model_output.get("context", "")
    
    prompt = FAITHFULNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        normalized = (score - 1) / 4
    except ValueError:
        normalized = None
    
    return {"faithfulness": normalized}


@weave.op()
def relevancy_score(question: str, reference: str, model_output: dict) -> dict:
    answer = model_output.get("answer", "")
    
    prompt = RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
        reference=reference
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    try:
        score = int(response.choices[0].message.content.strip())
        normalized = (score - 1) / 4
    except ValueError:
        normalized = None
    
    return {"relevancy": normalized}


evaluation = Evaluation(
    dataset=dataset,
    scorers=[
        context_precision,
        context_recall,
        faithfulness_score,
        relevancy_score
    ]
)

async def run_eval():
    rag_system = RAGSystem(top_k=2)
    results = await evaluation.evaluate(
        rag_system,
        __weave={"display_name": "rag-eval-top2"}
    )
    print("Done!")
    return results

import asyncio
asyncio.run(run_eval())

async def compare_configs():
    for k in [1, 2, 3]:
        rag_system = RAGSystem(top_k=k)
        await evaluation.evaluate(
            rag_system,
            __weave={"display_name": f"rag-eval-top{k}"}
        )
        print(f"Finished top_k={k}")

asyncio.run(compare_configs())