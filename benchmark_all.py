import subprocess
import requests
import time
import csv
import statistics
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT = """
Explain the difference between on-policy and off-policy reinforcement learning
in simple terms, then give one robotics example for each.
"""

TEMPERATURE = 0
WARMUP_RUNS = 1
MEASURED_RUNS = 3
OUTPUT_CSV = "ollama_benchmark_results.csv"


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout


def check_ollama():
    try:
        output = run_command(["ollama", "--version"])
        print("Ollama detected:")
        print(output.strip())
        print()
    except Exception as e:
        print("ERROR: Ollama CLI not found.")
        print(e)
        sys.exit(1)


def get_installed_models():
    """
    Read installed Ollama models from `ollama list`.
    """

    output = run_command(["ollama", "list"])
    lines = output.strip().splitlines()

    models = []

    for line in lines[1:]:
        if not line.strip():
            continue

        model_name = line.split()[0]
        models.append(model_name)

    return models


def test_server():
    try:
        r = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10
        )

        r.raise_for_status()

        print("Ollama server is running.")
        print()

    except Exception as e:
        print("ERROR: Cannot connect to Ollama server.")
        print()
        print("Start Ollama first.")
        print(e)
        sys.exit(1)


def run_model(model):
    """
    Run one inference request and return benchmark metrics.
    """

    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE
        }
    }

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=600
    )

    end = time.perf_counter()

    response.raise_for_status()

    data = response.json()

    wall_time = end - start

    prompt_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)

    load_duration = data.get("load_duration", 0) / 1e9
    prompt_duration = data.get("prompt_eval_duration", 0) / 1e9
    eval_duration = data.get("eval_duration", 0) / 1e9

    prompt_tps = (
        prompt_tokens / prompt_duration
        if prompt_duration > 0
        else 0
    )

    generation_tps = (
        output_tokens / eval_duration
        if eval_duration > 0
        else 0
    )

    return {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "load_time_s": load_duration,
        "prompt_time_s": prompt_duration,
        "generation_time_s": eval_duration,
        "wall_time_s": wall_time,
        "prompt_tps": prompt_tps,
        "generation_tps": generation_tps,
        "response": data.get("response", "")
    }


def unload_model(model):
    """
    Ask Ollama to release the model from memory.
    """

    try:
        payload = {
            "model": model,
            "keep_alive": 0
        }

        requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=60
        )

        time.sleep(2)

    except Exception:
        pass


def average(values):
    return statistics.mean(values) if values else 0


def benchmark_model(model):
    print()
    print("=" * 80)
    print(f"MODEL: {model}")
    print("=" * 80)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    for i in range(WARMUP_RUNS):
        print(f"Warm-up {i + 1}/{WARMUP_RUNS}")

        try:
            run_model(model)

        except Exception as e:
            print(f"Warm-up failed: {e}")
            return None

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    results = []

    for i in range(MEASURED_RUNS):

        print(f"Measured run {i + 1}/{MEASURED_RUNS}")

        try:

            result = run_model(model)

            results.append(result)

            print(
                f"  generation = "
                f"{result['generation_tps']:.2f} tok/s"
            )

            print(
                f"  prompt = "
                f"{result['prompt_tps']:.2f} tok/s"
            )

            print(
                f"  wall time = "
                f"{result['wall_time_s']:.2f} s"
            )

        except Exception as e:

            print(f"Run failed: {e}")

    if not results:
        return None

    summary = {
        "model": model,

        "prompt_tokens":
            round(average([
                x["prompt_tokens"]
                for x in results
            ]), 2),

        "output_tokens":
            round(average([
                x["output_tokens"]
                for x in results
            ]), 2),

        "avg_load_time_s":
            round(average([
                x["load_time_s"]
                for x in results
            ]), 4),

        "avg_prompt_time_s":
            round(average([
                x["prompt_time_s"]
                for x in results
            ]), 4),

        "avg_generation_time_s":
            round(average([
                x["generation_time_s"]
                for x in results
            ]), 4),

        "avg_wall_time_s":
            round(average([
                x["wall_time_s"]
                for x in results
            ]), 4),

        "avg_prompt_tps":
            round(average([
                x["prompt_tps"]
                for x in results
            ]), 2),

        "avg_generation_tps":
            round(average([
                x["generation_tps"]
                for x in results
            ]), 2),

        "runs_completed":
            len(results)
    }

    unload_model(model)

    return summary


def save_csv(results):

    if not results:
        print("No benchmark results to save.")
        return

    fieldnames = [
        "model",
        "prompt_tokens",
        "output_tokens",
        "avg_load_time_s",
        "avg_prompt_time_s",
        "avg_generation_time_s",
        "avg_wall_time_s",
        "avg_prompt_tps",
        "avg_generation_tps",
        "runs_completed"
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)


def print_summary(results):

    print()
    print("=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    print(
        f"{'Model':45}"
        f"{'Prompt tok/s':>16}"
        f"{'Generation tok/s':>20}"
        f"{'Wall time':>15}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{r['model'][:44]:45}"
            f"{r['avg_prompt_tps']:>16.2f}"
            f"{r['avg_generation_tps']:>20.2f}"
            f"{r['avg_wall_time_s']:>14.2f}s"
        )


def main():

    check_ollama()

    test_server()

    models = get_installed_models()

    if not models:
        print("No Ollama models installed.")
        sys.exit(0)

    print("Installed models:")

    for model in models:
        print(f"  - {model}")

    print()
    print(f"Total models: {len(models)}")

    results = []

    for index, model in enumerate(models, start=1):

        print()
        print(
            f"[{index}/{len(models)}] "
            f"Benchmarking {model}"
        )

        try:

            summary = benchmark_model(model)

            if summary:
                results.append(summary)

        except KeyboardInterrupt:

            print()
            print("Benchmark interrupted by user.")
            break

        except Exception as e:

            print(
                f"ERROR benchmarking {model}: {e}"
            )

    save_csv(results)

    print_summary(results)

    print()
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()