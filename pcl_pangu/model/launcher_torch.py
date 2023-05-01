import sys
import subprocess
import os
from loguru import logger
# logger = logging.getLogger(__name__)

def launch(nnodes=1, node_rank=0, nproc_per_node=1, master_addr="",
        master_port=29500, use_env=False, module=False, no_python=False,
        training_script_args=[],training_script='', **kwargs):

    # world size in terms of number of processes
    dist_world_size = nproc_per_node * nnodes

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = master_addr
    current_env["MASTER_PORT"] = str(master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    for local_rank in range(0, nproc_per_node):
        # each process's rank
        dist_rank = nproc_per_node * node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if module:
                cmd.append("-m")
        else:
            if not use_env:
                raise ValueError("When using the '--no_python' flag, you must also set the '--use_env' flag.")
            if module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(training_script)

        if not use_env:
            cmd.append("--local_rank={}".format(local_rank))

        cmd.extend(training_script_args)

        # os.chdir(python_work_dir)
        # logger.info(os.getcwd())
        # logger.info(os.path.abspath(__file__))
        logger.debug("Spawning process {} with command: {}".format(dist_rank, cmd))

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    launch()
