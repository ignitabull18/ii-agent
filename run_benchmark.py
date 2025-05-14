#!/usr/bin/env python3
"""
GAIA Dataset Evaluation Runner.

This script provides functionality to run evaluations on the GAIA dataset using the Agent system.
It integrates with the existing CLI infrastructure while adding GAIA-specific evaluation capabilities.
"""

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pvduy/duy/repos/ii-agent/ii_agent_vertex_ai_service_account.json"

import json
import argparse
from datetime import datetime
from pathlib import Path
import shutil
from threading import Lock
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download

from ii_agent.agents.anthropic_fc import AnthropicFC
from ii_agent.utils import WorkspaceManager
from ii_agent.llm import get_client
from ii_agent.llm.context_manager.standard import StandardContextManager
from ii_agent.llm.token_counter import TokenCounter
from utils import parse_common_args

# Global lock for thread-safe file appending
append_answer_lock = Lock()

BASE_TRACE_LOG_DIR = "trace_logs"
os.makedirs(BASE_TRACE_LOG_DIR, exist_ok=True)

BASE_WORKSPACE_DIR = "workspace"
os.makedirs(BASE_WORKSPACE_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments for GAIA evaluation."""
    parser = argparse.ArgumentParser(description="Run GAIA dataset evaluation")
    parser = parse_common_args(parser)
    
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which dataset split to evaluate on",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this evaluation run (used in output filename)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent evaluation tasks",
    )
    
    return parser.parse_args()

def load_gaia_dataset(split: str) -> Dataset:
    """Load the GAIA dataset, downloading if necessary."""
    if not os.path.exists("data/gaia"):
        snapshot_download(
            repo_id="gaia-benchmark/GAIA",
            repo_type="dataset",
            local_dir="data/gaia",
            ignore_patterns=[".gitattributes", "README.md"],
        )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_path"] = f"data/gaia/2023/{split}/" + row["file_name"]
        return row

    eval_ds = load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=split,
    )
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds

def append_answer(entry: dict, jsonl_file: str) -> None:
    """Append a single answer to the output JSONL file."""
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    assert jsonl_path.exists(), "File not found!"
    print("Answer exported to file:", jsonl_path.resolve())

def get_examples_to_answer(answers_file: str, eval_ds: Dataset) -> list[dict]:
    """Get list of examples that haven't been answered yet."""
    print(f"Loading answers from {answers_file}...")
    try:
        done_questions = pd.read_json(answers_file, lines=True)["question"].tolist()
        print(f"Found {len(done_questions)} previous results!")
    except Exception as e:
        print("Error when loading records: ", e)
        print("No usable records! ▶️ Starting new.")
        done_questions = []
    return [line for line in eval_ds.to_list() 
            if line["Question"] not in done_questions]
    

def answer_single_question(
    example: dict,
    answers_file: str,
    llm,
    context_manager,
    logger,
    trace_dir: str,
    workspace_dir: str,
) -> None:
    task_id = example["task_id"]
    question = example["Question"]
    print(f"Processing task {task_id}...")

    trace_dir = f"{trace_dir}/task_{task_id}"
    workspace_dir = f"{workspace_dir}/task_{task_id}"

    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    # dump json example to trace_dir
    with open(f"{trace_dir}/input.json", "w") as f:
        json.dump(example, f, indent=4)

    # Copy required files to workspace if they exist
    relative_file_path = None
    if example["file_name"]:
        file_path = example["file_path"]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.split(".")[-1]
        new_file_name = f"file.{file_extension}"
        user_uploads_dir = f"{workspace_dir}/user_uploads"
        os.makedirs(user_uploads_dir, exist_ok=True)

        new_file_path = os.path.join(user_uploads_dir, new_file_name)
        
        # Copy the file to upload_dir
        shutil.copy2(file_path, new_file_path)

        if not os.path.exists(new_file_path):
            raise FileNotFoundError(f"Copying file to workspace failed, file not found: {new_file_path}")
        
        if file_extension in ["jpg", "jpeg", "png", "gif", "webp"]:
            relative_file_path = new_file_path
        else:
            relative_file_path = f"user_uploads/{new_file_name}"

    # Create workspace manager for this question
    workspace_manager = WorkspaceManager(
        root=Path(workspace_dir),
        container_workspace=None,
    )

    # Create agent instance for this question
    agent = AnthropicFC(
        client=llm,
        workspace_manager=workspace_manager,
        logger_for_agent_logs=logger,
        context_manager=context_manager,
        max_output_tokens_per_turn=32768,
        max_turns=200,
    )

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        final_result = agent.run_agent(
            question,
            resume=False,
            log_dir=trace_dir,
            files=[relative_file_path] if relative_file_path else None
        )
        output = str(final_result)
        limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
        raised_exception = False
        exception = None

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        output = None
        limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # dump output to trace_dir
    with open(f"{trace_dir}/output.json", "w") as f:
        json.dump({
            "output": output,
            "limit_exceeded": limit_exceeded,
            "raised_exception": raised_exception,
            "exception": str(exception) if raised_exception else None
        }, f, indent=4)
        
    
    annotated_example = {
        "task_id": task_id,
        "question": question,
        "file_name": example["file_name"],
        "level": example["Level"],
        "ground_truth": example["Final answer"],
        "prediction": output,
        "Annotator Metadata": example["Annotator Metadata"],
        "limit_exceeded": limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "start_time": start_time,
        "end_time": end_time,
    }
    
    append_answer(annotated_example, answers_file)

def main():
    """Main entry point for GAIA evaluation."""
    args = parse_args()
    print(f"Starting GAIA evaluation with arguments: {args}")

    # Setup logging
    if os.path.exists(args.logs_path):
        os.remove(args.logs_path)
    logger = logging.getLogger("gaia_eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.FileHandler(args.logs_path))
    if not args.minimize_stdout_logs:
        logger.addHandler(logging.StreamHandler())
    else:
        logger.propagate = False

    # Initialize LLM client
    llm = get_client(
        "anthropic-direct",
        model_name="claude-3-7-sonnet@20250219",
        use_caching=False,
        project_id=args.project_id,
        region=args.region,
        thinking_tokens=2048
    )

    # Initialize token counter and context manager
    token_counter = TokenCounter()
    context_manager = StandardContextManager(
        token_counter=token_counter,
        logger=logger,
        token_budget=120_000,
    )

    # Load dataset and get tasks to run
    eval_ds = load_gaia_dataset(args.split)
    print("Number of tasks to run: ", len(eval_ds))

    answers_file = f"output/{args.split}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)

    trace_dir = f"{BASE_TRACE_LOG_DIR}/{args.run_name}"
    workspace_dir = f"{BASE_WORKSPACE_DIR}/{args.run_name}"
    os.makedirs(trace_dir, exist_ok=True)
    os.makedirs(workspace_dir, exist_ok=True)

    # Process tasks
    for example in tqdm(tasks_to_run, desc="Processing GAIA tasks"):
        answer_single_question(
            example,
            answers_file,
            llm,
            context_manager,
            logger,
            trace_dir,
            workspace_dir,
        )

    print("All GAIA tasks processed.")

if __name__ == "__main__":
    main() 