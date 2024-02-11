import os
from tqdm import tqdm
from checker import load_solutions
from utils import write_directory


if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=str, required=True)
    parser.add_argument("--eofs", nargs="+", type=str, default=[])
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--remove_prompt", action="store_true")
    parser.add_argument("--source_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument("--target_lang", required=True, type=str, choices=["C", "C++", "Java", "Python", "Go"])
    parser.add_argument(
        "--debug-task", type=str, help="Enter the task ID to only sanitize that task."
    )
    parser.add_argument("--rm-prefix-lines", nargs="+", type=str, help="Remove lines starting with these string.", default=[])
    args = parser.parse_args()

    EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }

    # make a new folder with "-sanitized" suffix
    is_folder = os.path.isdir(args.samples)
    target_path = pathlib.Path(args.samples)
    if not args.inplace:
        if is_folder:
            new_name = target_path.name + "-sanitized"
        else:
            new_name = target_path.name.replace(".jsonl", "-sanitized.jsonl")
        target_path = target_path.parent / new_name
    target_path = str(target_path)

    nsan = 0
    ntotal = 0

    new_solutions = []

    for solution in tqdm(load_solutions(args)):
        task_id = solution["task_id"]
        dbg_identifier = solution["_identifier"]
        if args.debug_task is not None and task_id != args.debug_task:
            continue

        ntotal += 1
        old_code = solution["solution"]

        # start to modify old_code | old_code should not be re-defined

        new_code = old_code

        if args.remove_prompt:
            new_code = new_code[new_code.find(f"{args.target_lang}:\n") + len(f"{args.target_lang}:\n"):]
            while new_code.strip().startswith(f"{args.target_lang}:"):
                new_code = new_code[new_code.find(f"{args.target_lang}:\n") + len(f"{args.target_lang}:\n"):]

        # basic handling of chat output
        new_code = new_code.replace(f"```{args.target_lang.lower()}", "").replace("```", "").strip()

        for prefix in args.rm_prefix_lines:
            new_code = "\n".join(
                [
                    line
                    for line in new_code.splitlines()
                    if not line.startswith(prefix)
                ]
            ).strip()

        for eof in args.eofs:
            new_code = new_code.split(eof)[0].strip()

        new_solutions.append(
            {
                "task_id": solution["task_id"],
                "solution": new_code.strip(),
            }
        )

    if is_folder:
        write_directory(target_path, new_solutions, ext=EXTENSIONS[args.target_lang])
