def cmd_mlx(args):
    """Run LISA training with MLX (Apple Silicon) via mlx_lm CLI."""
    import subprocess, json, tempfile, os, sys

    iters = args.train_steps if args.train_steps is not None else args.iters
    lora_rank = args.lora_rank or 4
    print(f"Training with LISA (MLX — Apple Silicon)")
    print(f"  Model: {args.model}")
    print(f"  Iterations: {iters}")
    print(f"  LoRA rank: {lora_rank}")

    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load
    except ImportError as e:
        print(f"  ERROR: MLX not available: {e}")
        print(f"  Install with: pip install mlx mlx-lm")
        return {"status": "error", "message": "MLX not installed"}

    # Verify model loads
    print(f"  Loading {args.model}...")
    try:
        model, tokenizer = mlx_load(args.model)
        del model, tokenizer  # Free memory
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {"status": "error", "message": str(e)}
    print(f"  Model verified OK")

    # Create temp dir for training data
    tmpdir = tempfile.mkdtemp(prefix="lisa_mlx_")
    data_path = os.path.join(tmpdir, "train.jsonl")

    # Prepare wikitext training data
    print(f"  Preparing training data...")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
        ds = ds.filter(lambda x: len(x.get("text", "").strip()) > 20)
        ds = ds.select(range(min(1000, len(ds))))
        with open(data_path, "w") as f:
            for item in ds:
                text = item.get("text", "").strip()
                if text:
                    f.write(json.dumps({"text": text}) + "\n")
        print(f"  Data: {len(ds)} samples")
    except Exception as e:
        print(f"  Dataset error: {e}. Using synthetic data.")
        with open(data_path, "w") as f:
            for i in range(500):
                f.write(json.dumps({"text": f"Training example {i}: the model learns from data"}) + "\n")

    # Build mlx_lm lora command
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train-data", f"{data_path}:jsonl",
        "--batch-size", str(args.batch_size),
        "--iters", str(iters),
        "--rank", str(lora_rank),
        "--steps-per-report", "10",
        "--max-seq-length", str(args.max_seq),
    ]

    print(f"  Running: python -m mlx_lm.lora --model {args.model} --iters {iters}...")
    t0 = __import__("time").time()
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    elapsed = __import__("time").time() - t0

    # Print output
    if result.stdout:
        print(f"\n{result.stdout[-1000:]}")
    if result.stderr:
        print(f"\nSTDERR: {result.stderr[-500:]}")

    if result.returncode == 0:
        # Find final loss
        final_loss = "unknown"
        for line in reversed(result.stdout.splitlines()):
            if "loss" in line.lower() and ":" in line:
                final_loss = line.strip()
                break
        print(f"\n  ✅ MLX training complete in {elapsed:.1f}s")
        return {"status": "success", "time": elapsed, "final_loss": final_loss}
    else:
        print(f"\n  ❌ MLX training failed (code {result.returncode})")
        return {"status": "error", "message": result.stderr[-500:]}


