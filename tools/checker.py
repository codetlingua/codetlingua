"""This file checks two things:
1. Is the LLMs codegen completed for each benchmark?
2. Warn the code that are not compilable (it could be some impl issues).
"""

import os
import ast
import traceback
import javalang
import subprocess

from termcolor import colored
from datasets import load_dataset
from utils import load_solutions


def syntax_check(code, target_lang, verbose=False):
    os.makedirs("temp_dir", exist_ok=True)

    if target_lang == "Python":
        try:
            ast.parse(code)
            return True
        except (SyntaxError, MemoryError):
            if verbose:
                traceback.print_exc()
            return False
        
    elif target_lang == "Java":
        try:
            javalang.parse.parse(code)
            return True
        except Exception:
            if verbose:
                traceback.print_exc()
            return False
        
    elif target_lang == "C":
        with open("temp_dir/temp.c", "w") as f:
            f.write(code)
        try:
            subprocess.run("gcc temp_dir/temp.c", check=True, capture_output=True, shell=True, timeout=10)
            os.remove("temp_dir/temp.c")
            return True
        except Exception:
            if verbose:
                traceback.print_exc()
            os.remove("temp_dir/temp.c")
            return False
        
    elif target_lang == "C++":
        with open("temp_dir/temp.cpp", "w") as f:
            f.write(code)
        try:
            subprocess.run("g++ temp_dir/temp.cpp", check=True, capture_output=True, shell=True, timeout=10)
            os.remove("temp_dir/temp.cpp")
            return True
        except Exception:
            if verbose:
                traceback.print_exc()
            os.remove("temp_dir/temp.cpp")
            return False

    elif target_lang == "Go":
        with open("temp_dir/temp.go", "w") as f:
            f.write(code)
        try:
            subprocess.run("go build temp_dir/temp.go", check=True, capture_output=True, shell=True, timeout=10)
            os.remove("temp_dir/temp.go")
            return True
        except Exception:
            if verbose:
                traceback.print_exc()
            os.remove("temp_dir/temp.go")
            return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["codenet", "avatar"]
    )
    parser.add_argument("--source_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--target_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--nsample", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    solutions = load_solutions(args)
    
    if args.dataset == "codenet":
        dataset = load_dataset("iidai/codenet")['train']
        dataset_name = "codenet"
    elif args.dataset == "avatar":
        dataset = load_dataset("iidai/avatar")['train']
        dataset_name = "avatar"

    id2solutions = {}
    for solution in solutions:
        task_id = solution["task_id"]
        if task_id not in id2solutions:
            id2solutions[task_id] = []
        if "solution" not in solution:
            assert "completion" in solution, "solution or completion must exist!"
            solution["solution"] = dataset[task_id]["code"]
        id2solutions[task_id].append(solution)

    nsample = max(args.nsample, max(len(v) for v in id2solutions.values()))
    print(colored("==============================", "blue"))
    print(colored(" ::: Checking completeness... ", "blue"))
    print(colored(" ::::: All tasks complete?    ", "blue"))
    ndone = 0

    task_ids = [x['id'] for x in dataset if x['language'] == args.source_lang]
    ntask = len(task_ids)
    for task_id in task_ids:
        if task_id not in id2solutions:
            print(colored(f" ⚠️ {task_id} is missing!", "red"))
            continue
        nfiles = len(id2solutions[task_id])
        if nfiles == nsample:
            ndone += 1
            continue

        print(
            colored(
                f" ⚠️ {task_id} only has {nfiles} samples! But {nsample} are expected.",
                "red",
            )
        )

    if ntask != ndone:
        ntbd = ntask - ndone
        print(colored(f" ::::: ⚠️ {ntbd}/{ntask} tasks incomplete!", "red"))
    else:
        print(colored(f" ::::: All {ntask} tasks complete!", "green"))

    print(colored("==============================", "blue"))
    print(colored(" ::: Checking compilation...  ", "blue"))
    print(colored(" ::::: All code compilable?   ", "blue"))
    ncode = 0
    nwrong = 0
    for task_id in task_ids:
        # task_id must exist
        if task_id not in id2solutions:
            continue

        for solution in id2solutions[task_id]:
            ncode += 1
            code = solution["solution"]
            dbg_identifier = solution["_identifier"]
            if code.strip() == "":
                print(colored(f" ⚠️ {dbg_identifier} is empty!", "red"))
                nwrong += 1
            elif not syntax_check(code, args.target_lang, args.verbose):
                print(colored(f" ⚠️ {dbg_identifier} is not compilable!", "red"))
                nwrong += 1
    if 0 != nwrong:
        print(colored(f" ::::: ⚠️ {nwrong}/{ncode} code are not compilable!", "red"))
        print(colored(f" ::::: ⚠️ {round((nwrong / ncode) * 100,2)}% of code are not compilable!", "red"))
    else:
        print(colored(f" ::::: All {ncode} code are compilable!", "green"))
