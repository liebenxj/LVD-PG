import multiprocessing as mp
import subprocess
import numpy as np
import torch
import os


class Worker(mp.Process):
    def __init__(self, pipe, do_work_func):
        super(Worker, self).__init__()

        self.pipe = pipe
        self.do_work_func = do_work_func

    def run(self):

        while True:

            command, args = self._recv_message()

            if command == "kill":
                return

            elif "execute" in command:
                if "cuda:" in command:
                    cuda = command.split("cuda:")[1]
                    command = f"CUDA_VISIBLE_DEVICES={cuda} {args}"
                    print(command)
                    process = subprocess.Popen(
                        command, shell = True, stdout = subprocess.PIPE
                    )
                    process.wait()
                # self.do_work_func(args)
                self._send_message("job_done")

    def _send_message(self, command, args = None):
        self.pipe.send((command, args))

    def _recv_message(self):
        self.pipe.poll(None) # wait until new message is received
        command, args = self.pipe.recv()

        return command, args


class JobManager():
    def __init__(self, job_args, devices, do_work_func = lambda x: None):
        self.do_work_func = do_work_func
        self.job_args = job_args
        self.num_workers = len(devices)
        self.devices = devices
        
        self.workers = []
        self.pipes = []
        self.worker_status = np.zeros([self.num_workers], dtype = np.int8)

        for _ in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            self.pipes.append(parent_pipe)

            worker = Worker(child_pipe, self.do_work_func)
            self.workers.append(worker)

    def start(self):
        for worker in self.workers:
            worker.start()

        worker_id = 0

        while len(self.job_args) > 0 or np.any(self.worker_status == 1):

            self.send_jobs()
            
            command, _ = self._recv_message_nonblocking(worker_id)
            if command is None:
                continue

            assert command == "job_done"
            self.worker_status[worker_id] = 0

            worker_id += 1
            if worker_id >= self.num_workers:
                worker_id = 0

        for worker_id in range(self.num_workers):
            self._send_message(worker_id, "kill")

    def send_jobs(self):
        for worker_id in range(self.num_workers):
            if len(self.job_args) > 0 and self.worker_status[worker_id] == 0:
                self.worker_status[worker_id] = 1
                cuda = self.devices[worker_id]
                self._send_message(worker_id, f"execute_cuda:{cuda}", self.job_args.pop())

    def _send_message(self, worker_idx, command, args = None):
        self.pipes[worker_idx].send((command, args))

    def _recv_message_nonblocking(self, worker_id):
        if not self.pipes[worker_id].poll():
            return None, None

        command, args = self.pipes[worker_id].recv()

        return command, args