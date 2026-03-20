"""
API Runner for Claude Constitution Culture Eval

Sends all prompts to the Anthropic API using the Messages Batch API for efficiency.
Falls back to sequential calls if batch API is unavailable.
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime
import anthropic

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "raw_responses"

# Model to evaluate
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 1.0  # For variance estimation


def load_prompts(path: Path = None) -> list[dict]:
    """Load prompts from JSONL file."""
    if path is None:
        path = PROMPTS_DIR / "all_prompts.jsonl"
    prompts = []
    with open(path) as f:
        for line in f:
            prompts.append(json.loads(line))
    return prompts


def run_sequential(prompts: list[dict], model: str = MODEL,
                   runs_per_prompt: int = 1, resume_from: str = None) -> list[dict]:
    """
    Run prompts sequentially via the Messages API.
    Saves results incrementally so we can resume on failure.
    """
    client = anthropic.Anthropic()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"responses_{timestamp}.jsonl"

    # Load existing results if resuming
    completed_ids = set()
    if resume_from:
        resume_path = RESULTS_DIR / resume_from
        if resume_path.exists():
            with open(resume_path) as f:
                for line in f:
                    r = json.loads(line)
                    completed_ids.add((r['prompt_id'], r['run']))
            output_path = resume_path
            print(f"Resuming from {resume_path}, {len(completed_ids)} already completed")

    total = len(prompts) * runs_per_prompt
    done = len(completed_ids)

    print(f"Running {total - done} remaining API calls ({total} total, {done} done)")
    print(f"Model: {model}")
    print(f"Saving to: {output_path}")

    with open(output_path, 'a') as f:
        for run in range(runs_per_prompt):
            for i, prompt in enumerate(prompts):
                prompt_id = prompt['prompt_id']

                if (prompt_id, run) in completed_ids:
                    continue

                try:
                    message = client.messages.create(
                        model=model,
                        max_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        system=prompt['system_prompt'],
                        messages=[
                            {"role": "user", "content": prompt['user_prompt']}
                        ]
                    )

                    response_text = message.content[0].text

                    result = {
                        'prompt_id': prompt_id,
                        'item_id': prompt['item_id'],
                        'domain': prompt['domain'],
                        'format': prompt['format'],
                        'condition': prompt['condition'],
                        'country': prompt['country'],
                        'topic': prompt['topic'],
                        'run': run,
                        'model': model,
                        'response': response_text,
                        'input_tokens': message.usage.input_tokens,
                        'output_tokens': message.usage.output_tokens,
                        'stop_reason': message.stop_reason,
                        'timestamp': datetime.now().isoformat(),
                    }

                    f.write(json.dumps(result) + '\n')
                    f.flush()

                    done += 1
                    if done % 50 == 0:
                        print(f"  Progress: {done}/{total} ({done/total*100:.1f}%)")

                except anthropic.RateLimitError:
                    print(f"  Rate limited at {done}/{total}, waiting 30s...")
                    time.sleep(30)
                    # Retry
                    try:
                        message = client.messages.create(
                            model=model,
                            max_tokens=MAX_TOKENS,
                            temperature=TEMPERATURE,
                            system=prompt['system_prompt'],
                            messages=[
                                {"role": "user", "content": prompt['user_prompt']}
                            ]
                        )
                        response_text = message.content[0].text
                        result = {
                            'prompt_id': prompt_id,
                            'item_id': prompt['item_id'],
                            'domain': prompt['domain'],
                            'format': prompt['format'],
                            'condition': prompt['condition'],
                            'country': prompt['country'],
                            'topic': prompt['topic'],
                            'run': run,
                            'model': model,
                            'response': response_text,
                            'input_tokens': message.usage.input_tokens,
                            'output_tokens': message.usage.output_tokens,
                            'stop_reason': message.stop_reason,
                            'timestamp': datetime.now().isoformat(),
                        }
                        f.write(json.dumps(result) + '\n')
                        f.flush()
                        done += 1
                    except Exception as e:
                        print(f"  ERROR on retry for {prompt_id}: {e}")

                except Exception as e:
                    print(f"  ERROR for {prompt_id}: {e}")

    print(f"\nDone! {done} responses saved to {output_path}")
    return output_path


def run_batch(prompts: list[dict], model: str = MODEL,
              runs_per_prompt: int = 1) -> str:
    """
    Run prompts via the Messages Batch API (50% cost savings).
    """
    client = anthropic.Anthropic()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Build batch requests
    requests = []
    for run in range(runs_per_prompt):
        for prompt in prompts:
            custom_id = f"{prompt['prompt_id']}_run{run}"
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": prompt['system_prompt'],
                    "messages": [
                        {"role": "user", "content": prompt['user_prompt']}
                    ]
                }
            })

    print(f"Submitting batch of {len(requests)} requests...")

    # Submit batch
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id
    print(f"Batch submitted: {batch_id}")
    print(f"Status: {batch.processing_status}")

    # Save batch ID for later retrieval
    batch_info = {
        'batch_id': batch_id,
        'n_requests': len(requests),
        'model': model,
        'submitted_at': datetime.now().isoformat(),
    }
    info_path = RESULTS_DIR / f"batch_{batch_id}.json"
    with open(info_path, 'w') as f:
        json.dump(batch_info, f, indent=2)

    print(f"Batch info saved to {info_path}")
    print(f"\nTo check status: python -c \"import anthropic; c=anthropic.Anthropic(); print(c.messages.batches.retrieve('{batch_id}').processing_status)\"")

    return batch_id


def retrieve_batch_results(batch_id: str) -> Path:
    """Retrieve and save results from a completed batch."""
    client = anthropic.Anthropic()

    # Check status
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Batch {batch_id} status: {batch.processing_status}")

    if batch.processing_status != "ended":
        print("Batch not yet complete. Try again later.")
        return None

    # Load prompt metadata for joining
    prompts = load_prompts()
    prompt_lookup = {p['prompt_id']: p for p in prompts}

    # Retrieve results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"batch_{batch_id}_results.jsonl"

    count = 0
    with open(output_path, 'w') as f:
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            prompt_id = custom_id.rsplit('_run', 1)[0]
            run = int(custom_id.rsplit('_run', 1)[1])

            prompt_meta = prompt_lookup.get(prompt_id, {})

            if result.result.type == "succeeded":
                message = result.result.message
                response_text = message.content[0].text if message.content else ""

                record = {
                    'prompt_id': prompt_id,
                    'item_id': prompt_meta.get('item_id'),
                    'domain': prompt_meta.get('domain'),
                    'format': prompt_meta.get('format'),
                    'condition': prompt_meta.get('condition'),
                    'country': prompt_meta.get('country'),
                    'topic': prompt_meta.get('topic'),
                    'run': run,
                    'model': message.model,
                    'response': response_text,
                    'input_tokens': message.usage.input_tokens,
                    'output_tokens': message.usage.output_tokens,
                    'stop_reason': message.stop_reason,
                }
            else:
                record = {
                    'prompt_id': prompt_id,
                    'run': run,
                    'error': str(result.result.type),
                }

            f.write(json.dumps(record) + '\n')
            count += 1

    print(f"Saved {count} results to {output_path}")
    return output_path


def estimate_cost(prompts: list[dict], runs_per_prompt: int = 1):
    """Estimate API cost before running."""
    n_calls = len(prompts) * runs_per_prompt

    # Estimate tokens
    avg_input = 200  # tokens
    avg_output = 300  # tokens (Format A ~20, Format B ~500, average ~300)

    total_input = n_calls * avg_input
    total_output = n_calls * avg_output

    # Sonnet pricing (per million tokens)
    input_cost = total_input / 1_000_000 * 3.0
    output_cost = total_output / 1_000_000 * 15.0

    # Batch pricing (50% discount)
    batch_input_cost = input_cost * 0.5
    batch_output_cost = output_cost * 0.5

    print(f"=== COST ESTIMATE ===")
    print(f"Total API calls: {n_calls}")
    print(f"Estimated tokens: {total_input:,} input, {total_output:,} output")
    print(f"\nSequential pricing:")
    print(f"  Input:  ${input_cost:.2f}")
    print(f"  Output: ${output_cost:.2f}")
    print(f"  Total:  ${input_cost + output_cost:.2f}")
    print(f"\nBatch pricing (50% off):")
    print(f"  Input:  ${batch_input_cost:.2f}")
    print(f"  Output: ${batch_output_cost:.2f}")
    print(f"  Total:  ${batch_input_cost + batch_output_cost:.2f}")


if __name__ == "__main__":
    import sys

    prompts = load_prompts()

    # Support --format A or --format B flag to filter prompts
    # Support --no-system to strip system prompts (ablation study)
    format_filter = None
    no_system = False
    model_override = None
    filtered_args = []
    for i, arg in enumerate(sys.argv):
        if arg == "--format" and i + 1 < len(sys.argv):
            format_filter = sys.argv[i + 1]
        elif arg == "--model" and i + 1 < len(sys.argv):
            model_override = sys.argv[i + 1]
        elif arg == "--no-system":
            no_system = True
        elif i > 0 and sys.argv[i - 1] not in ("--format", "--model"):
            filtered_args.append(arg)
    sys.argv = [sys.argv[0]] + filtered_args

    if model_override:
        print(f"Model override: {model_override}")

    if format_filter:
        prompts = [p for p in prompts if p['format'] == format_filter]
        print(f"Filtered to Format {format_filter}: {len(prompts)} prompts")

    if no_system:
        for p in prompts:
            p['system_prompt'] = ""
            p['condition'] = p.get('condition', '') + '_no_system'
        print(f"Stripped all system prompts (ablation mode)")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "estimate":
            runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            estimate_cost(prompts, runs)

        elif command == "run":
            runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            run_sequential(prompts, model=model_override or MODEL, runs_per_prompt=runs)

        elif command == "batch":
            runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
            run_batch(prompts, runs_per_prompt=runs)

        elif command == "retrieve":
            batch_id = sys.argv[2]
            retrieve_batch_results(batch_id)

        elif command == "resume":
            filename = sys.argv[2]
            runs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            run_sequential(prompts, runs_per_prompt=runs, resume_from=filename)

        else:
            print(f"Unknown command: {command}")
            print("Usage: python api_runner.py [estimate|run|batch|retrieve|resume] [args]")
    else:
        estimate_cost(prompts)
