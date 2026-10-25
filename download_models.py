import subprocess
import sys
import time


# ============================================================
# RTX 4060 8 GB
#
# Only variants that are practical for FULL GPU residency.
#
# "standard" is included because you requested it, although
# Ollama's default tag is usually already the Q4 model.
# ============================================================

MODELS = [

    # ========================================================
    # Models suitable for RTX 4060 8 GB
    # ========================================================

    # --------------------------------------------------------
    # Qwen3 4B
    # --------------------------------------------------------
    "qwen3:4b",
    "qwen3:4b-q4_K_M",
    "qwen3:4b-q8_0",

    # --------------------------------------------------------
    # Qwen3 8B
    # Q8 excluded because ~8.9 GB
    # --------------------------------------------------------
    "qwen3:8b",
    "qwen3:8b-q4_K_M",

    # --------------------------------------------------------
    # Gemma 3 4B
    # --------------------------------------------------------
    "gemma3:4b",
    "gemma3:4b-it-q4_K_M",
    "gemma3:4b-it-q8_0",

    # --------------------------------------------------------
    # DeepSeek-R1 Distill 7B
    # --------------------------------------------------------
    "deepseek-r1:7b",
    "deepseek-r1:7b-qwen-distill-q4_K_M",

    # --------------------------------------------------------
    # DeepSeek-R1 0528 Qwen3 8B
    # --------------------------------------------------------
    "deepseek-r1:8b",
    "deepseek-r1:8b-0528-qwen3-q4_K_M",


    # ========================================================
    # OVERSIZED MODELS
    #
    # These are intentionally included for testing CPU/RAM
    # offloading on RTX 4060 8 GB.
    # ========================================================

    # --------------------------------------------------------
    # Qwen3 14B
    #
    # Q4 ~9.3 GB
    # Q8 ~16 GB
    # --------------------------------------------------------
    "qwen3:14b-q4_K_M",
    "qwen3:14b-q8_0",

    # --------------------------------------------------------
    # Gemma 3 12B
    #
    # Q4 ~8.1 GB
    # Q8 ~13 GB
    # --------------------------------------------------------
    "gemma3:12b-it-q4_K_M",
    "gemma3:12b-it-q8_0",

    # --------------------------------------------------------
    # DeepSeek-R1 Distill Qwen 14B
    #
    # Q4 ~9 GB
    # Q8 ~16 GB
    # --------------------------------------------------------
    "deepseek-r1:14b-qwen-distill-q4_K_M",
    "deepseek-r1:14b-qwen-distill-q8_0",
]


def run_command(command):
    """Run a command and return stdout."""

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())

    return result.stdout


def check_ollama():
    """Verify that Ollama CLI is installed."""

    try:
        output = run_command(["ollama", "--version"])

        print("Ollama detected:")
        print(output.strip())
        print()

    except Exception as e:

        print("ERROR: Ollama was not found.")
        print()
        print("Make sure Ollama is installed and available in PATH.")
        print()
        print(e)

        sys.exit(1)


def get_installed_models():
    """Return currently installed Ollama model tags."""

    try:
        output = run_command(["ollama", "list"])

        installed = set()

        lines = output.splitlines()

        # First line is header
        for line in lines[1:]:

            if not line.strip():
                continue

            model_name = line.split()[0]

            installed.add(model_name)

        return installed

    except Exception as e:

        print(f"Warning: could not read installed models: {e}")

        return set()


def pull_model(model):
    """Pull one model and display Ollama's progress bar."""

    print()
    print("=" * 72)
    print(f"Downloading: {model}")
    print("=" * 72)

    start = time.perf_counter()

    # Do not capture output.
    # This lets Ollama show its normal download progress.
    result = subprocess.run(
        ["ollama", "pull", model]
    )

    elapsed = time.perf_counter() - start

    if result.returncode == 0:

        print()
        print(f"SUCCESS: {model}")
        print(f"Time   : {elapsed / 60:.2f} min")

        return True

    else:

        print()
        print(f"FAILED: {model}")

        return False


def main():

    check_ollama()

    installed = get_installed_models()

    print("=" * 72)
    print("RTX 4060 8 GB LOCAL-LLM MODEL DOWNLOADER")
    print("=" * 72)

    print()
    print("Models requested:")

    for model in MODELS:
        print(f"  {model}")

    print()
    print(f"Total tags: {len(MODELS)}")

    successes = []
    skipped = []
    failures = []

    for index, model in enumerate(MODELS, start=1):

        print()
        print("#" * 72)
        print(f"[{index}/{len(MODELS)}] {model}")
        print("#" * 72)

        if model in installed:

            print("Already installed -> SKIP")

            skipped.append(model)

            continue

        try:

            success = pull_model(model)

            if success:
                successes.append(model)
            else:
                failures.append(model)

        except KeyboardInterrupt:

            print()
            print("Download interrupted by user.")

            break

        except Exception as e:

            print()
            print(f"ERROR: {e}")

            failures.append(model)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("DOWNLOAD SUMMARY")
    print("=" * 72)

    print(f"Successful : {len(successes)}")
    print(f"Skipped    : {len(skipped)}")
    print(f"Failed     : {len(failures)}")

    if successes:

        print()
        print("Downloaded:")

        for model in successes:
            print(f"  + {model}")

    if skipped:

        print()
        print("Already installed:")

        for model in skipped:
            print(f"  = {model}")

    if failures:

        print()
        print("Failed:")

        for model in failures:
            print(f"  ! {model}")

    print()
    print("=" * 72)
    print("OLLAMA LIST")
    print("=" * 72)

    subprocess.run(
        ["ollama", "list"]
    )


if __name__ == "__main__":
    main()