"""
Prompt Versioning 101 - Complete Tutorial Script
================================================

Run this script to:
1. Create three prompt versions in Weave
2. Publish an evaluation dataset
3. Run evaluations comparing all versions
4. Create a leaderboard

After running, you can capture screenshots from the Weave UI:
- Assets → Prompts (shows three versions)
- Evals view (shows evaluation runs)
- Leaders view (shows leaderboard comparison)
- Compare view (diff between prompt versions)

Requirements:
    pip install weave openai python-dotenv

Setup:
    Create a .env file with:
    OPENAI_API_KEY=your-key-here
    
    Then run:
    wandb login
"""

import weave
from weave import Dataset, Evaluation, Model
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref
from openai import OpenAI
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI()

# Initialize Weave - change this to your entity/project
WEAVE_PROJECT = "prompt-versioning-demo"

def create_prompts():
    """Step 1 & 2: Create three prompt versions"""
    
    print("\n" + "="*50)
    print("STEP 1 & 2: Creating prompt versions")
    print("="*50)
    
    # v1: Minimal baseline
    prompt_v1 = weave.StringPrompt(
        """Classify this customer support message into one category:
- billing: Payment issues, invoices, refunds, subscription changes
- technical: Bugs, errors, how-to questions, feature requests
- account: Login issues, password resets, profile changes
- general: Everything else

Message: {message}

Category:"""
    )
    ref_v1 = weave.publish(prompt_v1, name="support_classifier")
    print(f"✓ Created v1 (minimal): {ref_v1.uri()}")
    
    # v2: With examples
    prompt_v2 = weave.StringPrompt(
        """Classify this customer support message into exactly one category.

Categories:
- billing: Payment issues, invoices, refunds, subscription changes
  Example: "I was charged twice this month"
- technical: Bugs, errors, how-to questions, feature requests  
  Example: "The app crashes when I try to upload a file"
- account: Login issues, password resets, profile changes
  Example: "I forgot my password and the reset email never arrived"
- general: Greetings, feedback, other
  Example: "What are your business hours?"

Message: {message}

Respond with only the category name."""
    )
    ref_v2 = weave.publish(prompt_v2, name="support_classifier")
    print(f"✓ Created v2 (with examples): {ref_v2.uri()}")
    
    # v3: Chain of thought
    prompt_v3 = weave.StringPrompt(
        """You are a customer support classifier.

Categories:
- billing: Payments, charges, invoices, refunds, subscriptions
- technical: Bugs, errors, feature questions, how-to help
- account: Login, password, profile settings
- general: Greetings, feedback, anything else

Message: {message}

Think step by step:
1. What is the customer asking about?
2. Which category fits best?
3. If multiple could apply, pick the most specific.

Category:"""
    )
    ref_v3 = weave.publish(prompt_v3, name="support_classifier")
    print(f"✓ Created v3 (chain of thought): {ref_v3.uri()}")
    
    return ref_v1.uri(), ref_v2.uri(), ref_v3.uri()


def create_dataset():
    """Step 3: Create evaluation dataset"""
    
    print("\n" + "="*50)
    print("STEP 3: Creating evaluation dataset")
    print("="*50)
    
    eval_data = [
        {"message": "I was charged twice for my subscription this month", 
         "expected": "billing"},
        {"message": "The app keeps crashing when I try to upload photos", 
         "expected": "technical"},
        {"message": "I can't log in, it says my password is wrong but I know it's correct", 
         "expected": "account"},
        {"message": "What time do you close on weekends?", 
         "expected": "general"},
        {"message": "How do I cancel my subscription and get a refund?", 
         "expected": "billing"},
        {"message": "Is there a way to export my data to CSV?", 
         "expected": "technical"},
        {"message": "I need to update my email address on file", 
         "expected": "account"},
        {"message": "Your product is great, just wanted to say thanks!", 
         "expected": "general"},
        {"message": "My invoice shows the wrong company name", 
         "expected": "billing"},
        {"message": "The search feature isn't finding anything even when I know items exist", 
         "expected": "technical"},
        {"message": "Can I have two accounts with the same email?", 
         "expected": "account"},
        {"message": "Do you have any job openings?", 
         "expected": "general"},
    ]
    
    dataset = Dataset(name="support_classifier_eval", rows=eval_data)
    ref = weave.publish(dataset)
    print(f"✓ Created dataset with {len(eval_data)} examples: {ref.uri()}")
    
    return dataset


# Step 4: Classifier model
class SupportClassifier(Model):
    prompt_ref: str
    model_name: str = "gpt-4o-mini"
    
    @weave.op()
    def predict(self, message: str) -> str:
        prompt = weave.ref(self.prompt_ref).get()
        formatted = prompt.format(message=message)
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted}],
            temperature=0.0,
            max_tokens=20
        )
        
        result = response.choices[0].message.content.strip().lower()
        
        # Extract category from response
        for category in ["billing", "technical", "account", "general"]:
            if category in result:
                return category
        return result


# Step 5: Scoring function
@weave.op()
def accuracy_scorer(expected: str, output: str) -> dict:
    """Check if predicted category matches expected"""
    match = expected.lower().strip() == output.lower().strip()
    return {"correct": match}


async def run_evaluations(prompt_refs: dict, dataset: Dataset):
    """Step 6: Run evaluations for each prompt version"""
    
    print("\n" + "="*50)
    print("STEP 6: Running evaluations")
    print("="*50)
    
    evaluation = Evaluation(
        dataset=dataset,
        scorers=[accuracy_scorer]
    )
    
    results = {}
    for name, ref in prompt_refs.items():
        print(f"\n→ Evaluating {name}...")
        classifier = SupportClassifier(prompt_ref=ref)
        result = await evaluation.evaluate(
            classifier,
            __weave={"display_name": name}
        )
        results[name] = result
        
        # Print accuracy
        accuracy = result.get('accuracy_scorer', {}).get('correct', {}).get('true_fraction', 0)
        print(f"  Accuracy: {accuracy:.1%}")
    
    # Create leaderboard
    print("\n" + "="*50)
    print("Creating leaderboard")
    print("="*50)
    
    spec = leaderboard.Leaderboard(
        name="prompt_version_comparison",
        description="Comparing support classifier prompt versions by accuracy",
        columns=[
            leaderboard.LeaderboardColumn(
                evaluation_object_ref=get_ref(evaluation).uri(),
                scorer_name="accuracy_scorer",
                summary_metric_path="correct.true_fraction",
            ),
        ],
    )
    lb_ref = weave.publish(spec)
    print(f"✓ Created leaderboard: {lb_ref.uri()}")
    
    return results


def main():
    """Run the complete tutorial"""
    
    print("\n" + "="*60)
    print("  PROMPT VERSIONING 101 - TUTORIAL SCRIPT")
    print("="*60)
    
    # Initialize Weave
    print(f"\nInitializing Weave project: {WEAVE_PROJECT}")

    import wandb

    wandb.init(
        entity="rfeers-databites",
        project=WEAVE_PROJECT,
    )

    # Step 1 & 2: Create prompts
    ref_v1, ref_v2, ref_v3 = create_prompts()
    
    # Step 3: Create dataset
    dataset = create_dataset()
    
    # Step 6: Run evaluations
    prompt_refs = {
        "v1_minimal": ref_v1,
        "v2_examples": ref_v2,
        "v3_chain_of_thought": ref_v3,
    }
    
    results = asyncio.run(run_evaluations(prompt_refs, dataset))
    
    # Summary
    print("\n" + "="*60)
    print("  COMPLETE!")
    print("="*60)
    print(f"""
Screenshots to capture in the Weave UI:

1. Assets → Prompts view
   Shows all three versions of 'support_classifier'
   
2. Click on 'support_classifier' → Compare versions
   Shows diff between any two prompt versions
   
3. Evals view
   Shows the three evaluation runs with accuracy scores
   
4. Leaders view
   Shows the leaderboard comparing all versions

Open your project at:
https://wandb.ai/YOUR_ENTITY/{WEAVE_PROJECT}/weave
""")


if __name__ == "__main__":
    main()