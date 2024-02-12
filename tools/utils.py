import os
import re
from os import PathLike
from typing import Dict, Iterable, List, Optional, Union, Any, Tuple
import multiprocessing
from multiprocessing import Value
import numpy as np
import subprocess
from subprocess import Popen, PIPE
import itertools
import time


EXTENSIONS = { "C": ".c", "C++": ".cpp", "Java": ".java", "Python": ".py", "Go": ".go" }

SUCCESS = "success"
RUNTIME_FAILED = "runtime_failed"
COMPILE_FAILED = "compile_failed"
INFINTIE_LOOP = "infinite_loop"
TEST_FAILED = "test_failed"

_SUCCESS = 0
_RUNTIME_FAILED = 1
_COMPILE_FAILED = 2
_INFINTIE_LOOP = 3
_TEST_FAILED = 4
_UNKNOWN = 5

_mapping = {_SUCCESS: SUCCESS, _RUNTIME_FAILED: RUNTIME_FAILED, _COMPILE_FAILED: COMPILE_FAILED, _INFINTIE_LOOP: INFINTIE_LOOP, _TEST_FAILED: TEST_FAILED, _UNKNOWN: None}


def write_directory(directory: PathLike, data: Iterable[Dict], ext: str):
    os.makedirs(directory, exist_ok=True)
    counters = {}
    for sample in data:
        assert "solution" in sample, "Samples must come with `solution` field!"
        task_id = sample["task_id"].replace("/", "_")
        task_dir = os.path.join(directory, task_id)
        os.makedirs(task_dir, exist_ok=True)
        if task_id not in counters:
            counters[task_id] = 0
        sample_id = counters[task_id]
        with open(os.path.join(task_dir, f"{sample_id}{ext}"), "w") as f:
            f.write(sample["solution"])
        counters[task_id] += 1


def load_solutions(args) -> List[Dict]:
    for task_id in os.listdir(args.samples):
        task_path = os.path.join(args.samples, task_id)
        if not os.path.isdir(task_path):
            continue

        for solution_id in os.listdir(task_path):
            solution_path = os.path.join(task_path, solution_id)
            if os.path.isfile(solution_path) and solution_path.endswith(f"{EXTENSIONS[args.target_lang]}"):
                with open(solution_path, "r") as f:
                    completion = f.read()
                yield {
                    "_identifier": solution_path,
                    "_path": solution_path,
                    "task_id": task_id,
                    "solution": completion,
                }


def exec_sample(
    problem: Dict[str, Any],
    code: str,
    target_lang: str,
    completion_id: int,
    stat: Value,
):
    import shutil

    os.makedirs(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}", exist_ok=True)

    if target_lang =="Python":

        with open(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.py", "w") as f:
            f.write(code)

        try:
            subprocess.run(f"python3 -m py_compile temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.py", check=True, capture_output=True, shell=True)

            test_io = problem['test_IO']
            for i in range(len(test_io)):
                f_in = test_io[i]['input']
                f_out = test_io[i]['output']
                
                p = Popen(f"python3 temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.py", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
                
                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    stat.value = _INFINTIE_LOOP
                    break
                
                try:
                    if float(stdout.decode())%1 == 0:
                        stdout = str(int(float(stdout.decode())))
                        f_out = str(int(float(f_out)))
                    else:
                        # find how many decimal points are there in the output
                        stdout_temp = stdout.decode().strip()
                        f_out_temp = f_out.strip()
                        f_out_total_dec_points = len(f_out_temp.split(".")[1])
                        stdout_total_dec_points = len(stdout_temp.split(".")[1])
                        min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                        stdout = str(round(float(stdout.decode()), min_dec_points))
                        f_out = str(round(float(f_out), min_dec_points))

                except:
                    try: # if stdout is already decoded as String, then pass
                        stdout = stdout.decode()
                    except:
                        pass
                
                if(stdout.strip()==f_out.strip()):
                    stat.value = _SUCCESS
                    continue

                else:
                    if stderr_data.decode()=='':
                        stat.value = _TEST_FAILED
                    else:
                        stat.value = _RUNTIME_FAILED
                    break

        except Exception as e:
            stat.value = _COMPILE_FAILED
    
        shutil.rmtree(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}")
    
    elif target_lang =="Java":

        with open(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.java", "w") as f:
            pattern = re.compile(r'\bclass\s+\w+')
            code = re.sub(pattern, f'class {problem["id"]}', code)
            f.write(code)
        
        try:
            subprocess.run(f"javac temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.java", check=True, capture_output=True, shell=True, timeout=30)

            test_io = problem['test_IO']
            for i in range(len(test_io)):
                f_in = test_io[i]['input']
                f_out = test_io[i]['output']

                p = Popen(f"java {problem['id']}", cwd=f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    stat.value = _INFINTIE_LOOP
                    break

                try:
                    if float(stdout.decode())%1 == 0:
                        stdout = str(int(float(stdout.decode())))
                        f_out = str(int(float(f_out)))
                    else:
                        # find how many decimal points are there in the output
                        stdout_temp = stdout.decode().strip()
                        f_out_temp = f_out.strip()
                        f_out_total_dec_points = len(f_out_temp.split(".")[1])
                        stdout_total_dec_points = len(stdout_temp.split(".")[1])
                        min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                        stdout = str(round(float(stdout.decode()), min_dec_points))
                        f_out = str(round(float(f_out), min_dec_points))

                except:
                    try: # if stdout is already decoded as String, then pass
                        stdout = stdout.decode()
                    except:
                        pass
                
                if(stdout.strip()==f_out.strip()):
                    stat.value = _SUCCESS
                    continue

                else:
                    if stderr_data.decode()=='':
                        stat.value = _TEST_FAILED
                    else:
                        stat.value = _RUNTIME_FAILED
                    break
        
        except Exception as e:
            stat.value = _COMPILE_FAILED
        
        shutil.rmtree(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}")
    
    elif target_lang =="C":

        with open(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.c", "w") as f:
            f.write(code)
        
        try:
            subprocess.run(f"gcc temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.c", check=True, capture_output=True, shell=True, timeout=30)

            test_io = problem['test_IO']
            for i in range(len(test_io)):
                f_in = test_io[i]['input']
                f_out = test_io[i]['output']

                p = Popen(f"./a.out", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    stat.value = _INFINTIE_LOOP
                    break

                try:
                    if float(stdout.decode())%1 == 0:
                        stdout = str(int(float(stdout.decode())))
                        f_out = str(int(float(f_out)))
                    else:
                        # find how many decimal points are there in the output
                        stdout_temp = stdout.decode().strip()
                        f_out_temp = f_out.strip()
                        f_out_total_dec_points = len(f_out_temp.split(".")[1])
                        stdout_total_dec_points = len(stdout_temp.split(".")[1])
                        min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                        stdout = str(round(float(stdout.decode()), min_dec_points))
                        f_out = str(round(float(f_out), min_dec_points))

                except:
                    try: # if stdout is already decoded as String, then pass
                        stdout = stdout.decode()
                    except:
                        pass
                
                if(stdout.strip()==f_out.strip()):
                    stat.value = _SUCCESS
                    continue

                else:
                    if stderr_data.decode()=='':
                        stat.value = _TEST_FAILED
                    else:
                        stat.value = _RUNTIME_FAILED
                    break
        
        except Exception as e:
            stat.value = _COMPILE_FAILED
        
        shutil.rmtree(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}")

    elif target_lang =="C++":

        with open(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.cpp", "w") as f:
            f.write(code)
        
        try:
            subprocess.run(f"g++ -o exec_output temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.cpp", check=True, capture_output=True, shell=True, timeout=30)

            test_io = problem['test_IO']
            for i in range(len(test_io)):
                f_in = test_io[i]['input']
                f_out = test_io[i]['output']

                p = Popen(f"./exec_output", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    stat.value = _INFINTIE_LOOP
                    break

                try:
                    if float(stdout.decode())%1 == 0:
                        stdout = str(int(float(stdout.decode())))
                        f_out = str(int(float(f_out)))
                    else:
                        # find how many decimal points are there in the output
                        stdout_temp = stdout.decode().strip()
                        f_out_temp = f_out.strip()
                        f_out_total_dec_points = len(f_out_temp.split(".")[1])
                        stdout_total_dec_points = len(stdout_temp.split(".")[1])
                        min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                        stdout = str(round(float(stdout.decode()), min_dec_points))
                        f_out = str(round(float(f_out), min_dec_points))

                except:
                    try: # if stdout is already decoded as String, then pass
                        stdout = stdout.decode()
                    except:
                        pass
                
                if(stdout.strip()==f_out.strip()):
                    stat.value = _SUCCESS
                    continue

                else:
                    if stderr_data.decode()=='':
                        stat.value = _TEST_FAILED
                    else:
                        stat.value = _RUNTIME_FAILED
                    break
        
        except Exception as e:
            stat.value = _COMPILE_FAILED
        
        shutil.rmtree(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}")

    elif target_lang =="Go":

        with open(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.go", "w") as f:
            f.write(code)
        
        try:
            subprocess.run(f"go build temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}/{problem['id']}.go", check=True, capture_output=True, shell=True, timeout=30)

            test_io = problem['test_IO']
            for i in range(len(test_io)):
                f_in = test_io[i]['input']
                f_out = test_io[i]['output']

                p = Popen(f"./{problem['id']}", stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
                
                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
                except subprocess.TimeoutExpired:
                    stat.value = _INFINTIE_LOOP
                    break

                try:
                    if float(stdout.decode())%1 == 0:
                        stdout = str(int(float(stdout.decode())))
                        f_out = str(int(float(f_out)))
                    else:
                        # find how many decimal points are there in the output
                        stdout_temp = stdout.decode().strip()
                        f_out_temp = f_out.strip()
                        f_out_total_dec_points = len(f_out_temp.split(".")[1])
                        stdout_total_dec_points = len(stdout_temp.split(".")[1])
                        min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                        stdout = str(round(float(stdout.decode()), min_dec_points))
                        f_out = str(round(float(f_out), min_dec_points))

                except:
                    try: # if stdout is already decoded as String, then pass
                        stdout = stdout.decode()
                    except:
                        pass
                
                if(stdout.strip()==f_out.strip()):
                    stat.value = _SUCCESS
                    continue

                else:
                    if stderr_data.decode()=='':
                        stat.value = _TEST_FAILED
                    else:
                        stat.value = _RUNTIME_FAILED
                    break
        
        except Exception as e:
            stat.value = _COMPILE_FAILED
        
        shutil.rmtree(f"temp_dir/{problem['id']}-{problem['language']}-{target_lang}-{completion_id}")


def untrusted_check(
    problem: Dict[str, Any],
    code: str,
    target_lang: str,
    completion_id: int,
) -> Tuple[str, np.ndarray]:

    # shared memory objects
    stat = Value("i", _UNKNOWN)

    p = multiprocessing.Process(
        target=exec_sample,
        args=(
            problem,
            code,
            target_lang,
            completion_id,
            stat,
        ),
    )

    p.start()
    p.join(100)

    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    return _mapping[stat.value]


def check_correctness(
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    identifier=None,
    args=None,
) -> Dict[str, Union[int, Optional[Tuple[str, List[bool]]]]]:

    ret = {
        "completion_id": completion_id,
        "task_id": problem["id"],
        "_identifier": identifier,
    }
    ret["base"] = untrusted_check(
        problem,
        solution,
        args.target_lang,
        completion_id,
    )

    return ret


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_problem(problems, task_id):
    for p in problems:
        if p["id"] == task_id:
            return p
    raise ValueError(f"Cannot find problem {task_id}")
