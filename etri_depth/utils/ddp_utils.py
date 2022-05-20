import builtins
import datetime
import errno
import os
import time
from collections import defaultdict, deque

import torch
import torch.distributed as dist

from .utils import StaticPrinter


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = reduce_across_processes([self.count, self.total])
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.global_avg}")
        return " ".join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=""):
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        MB = 1024.0 * 1024.0

        sp = StaticPrinter()
        for step, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if (step + 1) % print_freq == 0:
                sp.reset()
                sp.print(f"{header} [{step+1}/{len(iterable)}]")
                sp.print(f"iter_time: {iter_time}, data_time: {data_time}")
                sp.print(f"max mem: {torch.cuda.max_memory_allocated() / MB:.0f}")
                for name, meter in self.meters.items():
                    sp.print(f"{name}: {str(meter)}")
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """

    builtin_print = builtins.print

    def print_new(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print_new


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


# from https://github.com/TRI-ML/packnet-sfm
def on_rank_0(func):
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            func(*args, **kwargs)

    return wrapper


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def get_local_batch_size(global_batch_size: int):
    if not is_dist_avail_and_initialized():
        return global_batch_size

    assert global_batch_size >= dist.get_world_size()

    local_batch_size = global_batch_size // dist.get_world_size()
    if dist.get_rank() < global_batch_size % dist.get_world_size():
        local_batch_size += 1
    return local_batch_size


def is_main_process():
    return get_rank() == 0


def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        print("Not using distributed mode")
        return

    torch.cuda.set_device(local_rank)
    print(f"| distributed init (rank {rank})", flush=True)
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)


def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    dist.barrier()
    dist.all_reduce(t)
    return t
