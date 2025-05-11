#!/usr/bin/env python3
"""
GAIA Dataset Evaluation Runner.

This script provides functionality to run evaluations on the GAIA dataset using the Agent system.
It integrates with the existing CLI infrastructure while adding GAIA-specific evaluation capabilities.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import textwrap
from threading import Lock
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
import uuid

from ii_agent.agents.anthropic_fc import AnthropicFC
from ii_agent.utils import WorkspaceManager
from ii_agent.llm import get_client
from ii_agent.llm.context_manager.standard import StandardContextManager
from ii_agent.llm.token_counter import TokenCounter
from ii_agent.tools import TextInspectorTool
from utils import parse_common_args
from ii_agent.tools.gaia.image_qa import ImageQATool
from ii_agent.tools.gaia.doc_qa import DocQATool

# Global lock for thread-safe file appending
append_answer_lock = Lock()

def parse_args():
    """Parse command line arguments for GAIA evaluation."""
    parser = argparse.ArgumentParser(description="Run GAIA dataset evaluation")
    parser = parse_common_args(parser)
    
    # GAIA-specific arguments
    parser.add_argument(
        "--use-raw-dataset",
        action="store_true",
        help="Use raw GAIA dataset instead of annotated version",
    )
    parser.add_argument(
        "--set-to-run",
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
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index in the dataset (inclusive)",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Ending index in the dataset (exclusive). If not specified, runs until the end of dataset",
    )
    
    return parser.parse_args()

def load_gaia_dataset(use_raw_dataset: bool, set_to_run: str) -> Dataset:
    """Load the GAIA dataset, downloading if necessary."""
    if not os.path.exists("data/gaia"):
        if use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            # WARNING: this dataset is gated: make sure you visit the repo to require access
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir="data/gaia",
                ignore_patterns=[".gitattributes", "README.md"],
            )

    def preprocess_file_paths(row):
        if len(row["file_name"]) > 0:
            row["file_name"] = f"data/gaia/2023/{set_to_run}/" + row["file_name"]
        return row

    eval_ds = load_dataset(
        "data/gaia/GAIA.py",
        name="2023_all",
        split=set_to_run,
    )

    eval_ds = eval_ds.rename_columns({
        "Question": "question",
        "Final answer": "true_answer",
        "Level": "task"
    })
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
            if line["question"] not in done_questions]

def get_single_file_description(file_path: str, question: str, visual_inspection_tool, document_inspection_tool):
    file_extension = file_path.split(".")[-1]
    if file_extension in ["png", "jpg", "jpeg"]:
        file_description = f" - Attached image: {file_path}"
        file_description += (
            f"\n     -> Image description: {get_image_description(file_path, question, visual_inspection_tool)}"
        )
        return file_description
    elif file_extension in ["pdf", "xls", "xlsx", "docx", "doc", "xml"]:
        image_path = file_path.split(".")[0] + ".png"
        if os.path.exists(image_path):
            description = get_image_description(image_path, question, visual_inspection_tool)
            file_path = image_path
        else:
            description = get_document_description(file_path, question, document_inspection_tool)
        file_description = f" - Attached document: {file_path}"
        file_description += f"\n     -> File description: {description}"
        return file_description
    elif file_extension in ["mp3", "m4a", "wav"]:
        return f" - Attached audio: {file_path}"
    else:
        return f" - Attached file: {file_path}"

def get_zip_description(file_path: str, question: str, visual_inspection_tool: ImageQATool, document_inspection_tool: DocQATool) -> str:
    """Get descriptions of all files within a ZIP archive.
    
    Args:
        file_path: Path to the ZIP file
        question: The question to focus the descriptions on
        visual_inspection_tool: An instance of ImageQATool for processing images
        document_inspection_tool: An instance of DocQATool for processing documents
        
    Returns:
        str: A description of all relevant files in the ZIP archive
    """
    # Create a temporary directory to extract the ZIP contents
    folder_path = file_path.replace(".zip", "")
    os.makedirs(folder_path, exist_ok=True)
    
    try:
        # Extract the ZIP file
        shutil.unpack_archive(file_path, folder_path)
        
        # Process each file in the extracted directory
        file_descriptions = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = file.lower().split(".")[-1]
                
                # Skip hidden files and system files
                if file.startswith(".") or file.startswith("__"):
                    continue
                    
                # Process based on file type
                if file_ext in ["png", "jpg", "jpeg", "gif", "bmp"]:
                    try:
                        desc = get_image_description(file_path, question, visual_inspection_tool)
                        file_descriptions.append(f"Image file '{file}': {desc}")
                    except Exception as e:
                        file_descriptions.append(f"Image file '{file}': Error processing - {str(e)}")
                        
                elif file_ext in ["pdf", "docx", "doc", "txt", "xlsx", "xls", "xml"]:
                    try:
                        desc = get_document_description(file_path, question, document_inspection_tool)
                        file_descriptions.append(f"Document file '{file}': {desc}")
                    except Exception as e:
                        file_descriptions.append(f"Document file '{file}': Error processing - {str(e)}")
                        
                else:
                    file_descriptions.append(f"File '{file}': Unsupported file type")
        
        # Clean up the extracted directory
        shutil.rmtree(folder_path)
        
        if not file_descriptions:
            return "No processable files found in the ZIP archive."
            
        return "\n\n".join(file_descriptions)
        
    except Exception as e:
        # Clean up in case of error
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        return f"Error processing ZIP file: {str(e)}"

def get_image_description(file_name: str, question: str, visual_inspection_tool: ImageQATool) -> str:
    """Get a description of an image using the ImageQATool.
    
    Args:
        file_name: Path to the image file
        question: The question to focus the description on
        visual_inspection_tool: An instance of ImageQATool
        
    Returns:
        str: A description of the image focused on details relevant to the question
    """
    prompt = f"""Write a caption of 5 sentences for this image. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the image."""
    
    result = visual_inspection_tool.run_impl({
        "image_path": file_name,
        "question": prompt
    })
    return result.tool_output

def get_document_description(file_path: str, question: str, document_inspection_tool: DocQATool) -> str:
    """Get a description of a document using the DocQATool.
    
    Args:
        file_path: Path to the document file
        question: The question to focus the description on
        document_inspection_tool: An instance of DocQATool
        
    Returns:
        str: A description of the document focused on details relevant to the question
    """
    prompt = f"""Write a caption of 5 sentences for this document. Pay special attention to any details that might be useful for someone answering the following question:
{question}. But do not try to answer the question directly!
Do not add any information that is not present in the document."""
    
    result = document_inspection_tool.run_impl({
        "file_path": file_path,
        "question": prompt
    })
    return result.tool_output

def answer_single_question(
    example: dict,
    answers_file: str,
    visual_inspection_tool: ImageQATool,
    document_inspection_tool: DocQATool,
    logger: logging.Logger,
    client,
    context_manager,
    container_workspace: bool,
    needs_permission: bool,
    docker_container_id: str = None,
) -> None:
    """Process a single GAIA question using the agent."""
    # Create workspace using task_id
    task_id = example["task_id"]
    workspace_path = Path("workspace") / task_id
    workspace_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created workspace directory for task {task_id}: {workspace_path}")

    # Copy required files to workspace if they exist
    if example["file_name"]:
        source_file = Path(example["file_name"])
        if source_file.exists():
            # Create uploaded_files directory in workspace
            upload_dir = workspace_path / "uploaded_files"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the file to workspace
            dest_file = upload_dir / source_file.name
            shutil.copy2(source_file, dest_file)

            #check if same file name but with png extension exists (replace source_file extension with png)
            png_file = source_file.with_suffix(".png")
            if png_file.exists():
                #copy png file to workspace
                dest_png_file = upload_dir / png_file.name
                shutil.copy2(png_file, dest_png_file)
                logger.info(f"Copied file {png_file} to {dest_png_file}")

            logger.info(f"Copied file {source_file} to {dest_file}")
            
            # Update file path in example to point to workspace
            example["file_name"] = str(dest_file)
        else:
            logger.warning(f"Source file not found: {source_file}")

    # Create workspace manager for this question
    workspace_manager = WorkspaceManager(
        root=workspace_path,
        container_workspace=container_workspace
    )

    # Create agent instance for this question
    agent = AnthropicFC(
        client=client,
        workspace_manager=workspace_manager,
        logger_for_agent_logs=logger,
        context_manager=context_manager,
        max_output_tokens_per_turn=32768,
        max_turns=200,
        ask_user_permission=needs_permission,
        docker_container_id=docker_container_id,
    )

    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]

    if example["file_name"]:
        if ".zip" in example["file_name"]:
            prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
            prompt_use_files += get_zip_description(
                example["file_name"], 
                example["question"],
                visual_inspection_tool,
                document_inspection_tool
            )
        else:
            prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
            file_ext = example["file_name"].lower().split(".")[-1]
            if file_ext in ["png", "jpg", "jpeg", "gif", "bmp"]:
                desc = get_image_description(example["file_name"], example["question"], visual_inspection_tool)
                prompt_use_files += f"Image file: {desc}"
            else:
                desc = get_document_description(example["file_name"], example["question"], document_inspection_tool)
                prompt_use_files += f"Document file: {desc}"
        augmented_question += prompt_use_files

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # Run agent with question-specific workspace
        final_result = agent.run_agent(augmented_question, resume=True)
        
        output = str(final_result)
        intermediate_steps = [] #TODO: add this
        
        iteration_limit_exceeded = "Agent stopped due to iteration limit or time limit." in output
        raised_exception = False
        exception = None

    except Exception as e:
        logger.error(f"Error processing question: {e}")
        output = None
        intermediate_steps = []
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get token counts
    token_counts = 0 #TODO: add this
    
    annotated_example = {
        "agent_name": "anthropic-fc",
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example["task"],
        "task_id": task_id,
        "true_answer": example["true_answer"],
        "start_time": start_time,
        "end_time": end_time,
        "token_counts": token_counts,
        "workspace_id": task_id,
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
    client = get_client(
        "anthropic-direct",
        model_name="claude-3-7-sonnet@20250219",
        use_caching=False,
        project_id=args.project_id,
        region=args.region,
    )

    # Initialize token counter and context manager
    token_counter = TokenCounter()
    context_manager = StandardContextManager(
        token_counter=token_counter,
        logger=logger,
        token_budget=120_000,
    )

    # Initialize text inspection tools
    visual_inspection_tool = ImageQATool(workspace_manager=None, client=client)
    document_inspection_tool = DocQATool(workspace_manager=None, client=client, text_limit=100000)

    # Load dataset and get tasks to run
    eval_ds = load_gaia_dataset(args.use_raw_dataset, args.set_to_run)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())

    # Slice dataset based on start and end indices
    if args.end_index is None:
        args.end_index = len(eval_ds)
    if args.start_index < 0 or args.end_index > len(eval_ds) or args.start_index >= args.end_index:
        raise ValueError(f"Invalid range: start_index={args.start_index}, end_index={args.end_index}, dataset_size={len(eval_ds)}")
    
    eval_ds = eval_ds.select(range(args.start_index, args.end_index))
    print(f"Running evaluation on examples {args.start_index} to {args.end_index-1} (total: {len(eval_ds)} examples)")

    answers_file = f"output/{args.set_to_run}/{args.run_name}.jsonl"
    tasks_to_run = get_examples_to_answer(answers_file, eval_ds)

    # Process tasks
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(
                answer_single_question,
                example,
                answers_file,
                visual_inspection_tool,
                document_inspection_tool,
                logger,
                client,
                context_manager,
                args.use_container_workspace,
                args.needs_permission,
                args.docker_container_id,
            )
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Processing GAIA tasks"):
            f.result()

    print("All GAIA tasks processed.")

if __name__ == "__main__":
    main() 