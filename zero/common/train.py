import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.distributed import (all_reduce, get_rank, get_world_size, is_initialized)
from tqdm import tqdm

from common.gpt import NUM_EPOCHS


class AsyncMemoryMonitor:
    def __init__(self, rank, path='.', power=3):
        """
        Adapted from https://github.com/Tencent/PatrickStar/blob/master/patrickstar/core/memtracer/memtracer.py.
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.monitor_thread = None
        self.interval = 1 / (10**power)
        self.rank = rank
        self.file = path + f'/memory_rank_{rank}.log'

    def set_interval(self, power: int):
        self.interval = 1 / (10**power)

    def start(self):
        self.keep_measuring = True
        torch.cuda.reset_peak_memory_stats()
        self.monitor_thread = self.executor.submit(self._measure_usage)

    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        gpu_usage = self.monitor_thread.result()
        self.monitor_thread = None
        with open(self.file, 'a') as f:
            f.writelines(list(map(str, gpu_usage)))
        return gpu_usage

    def _measure_usage(self):
        gpu_usage = list()
        while self.keep_measuring:
            gpu_usage.append(torch.cuda.max_memory_allocated() / (1024 * 1024))  # MB
            torch.cuda.reset_peak_memory_stats()
            time.sleep(self.interval)

        return gpu_usage


def _print_log(msg):
    msg = f'{time.asctime()} > {msg}'
    rank = get_rank() if is_initialized() else 0
    log_path = os.environ['LOG'] if 'LOG' in os.environ else '.'
    log_file = log_path + f'/training_rank_{rank}.log'
    with open(log_file, 'a') as f:
        f.write(msg)
    if rank == 0:
        print(msg)


def _train(epoch, rank, world_size, train_dataloader, model, criterion, optimizer, lr_scheduler, scaler, mem_monitor,
           config):
    use_optimizer_backward = config['method'] in ['colossalai']
    if use_optimizer_backward and 'zero' in config:
        assert 'level' in config['zero'], 'Please provide zero level.'
        use_optimizer_backward = use_optimizer_backward and config['zero']['level'] < 3
    use_integrated_backward = config['method'] in ['deepspeed', 'patrickstar']
    use_integrated_step = config['method'] in ['deepspeed']
    use_autocast = config['method'] in ['torch'] and 'mixed_precision' in config
    if use_autocast and 'enabled' in config['mixed_precision']:
        use_autocast = use_autocast and config['mixed_precision']['enabled']
    clip_grad_norm = config['gradient_clipping'] if 'gradient_clipping' in config else 0.
    use_integraded_clip_grad = config['method'] in ['fairscale']

    model.train()

    progress = range(len(train_dataloader))
    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Train]")

    train_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_tokens = torch.zeros(()).to(torch.int).to(rank)
    data_iter = iter(train_dataloader)

    if mem_monitor is not None:
        mem_monitor.start()

    for _ in progress:
        fwd_start = time.time()

        optimizer.zero_grad()

        batch = next(data_iter)
        input_ids = batch['input_ids'].to(rank)
        attention_mask = batch['attention_mask'].to(rank)
        labels = batch['labels'].to(rank)
        batch_tokens = labels.numel()
        if use_autocast:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(outputs, labels)
        train_loss += loss

        fwd_end = time.time()

        bwd_start = time.time()

        if use_integrated_backward:  # deepspeed & patrickstar style
            model.backward(loss)
            if use_integrated_step:
                model.step()  # deepspeed style
            else:
                optimizer.step()  # patrickstar style
                lr_scheduler.step()

        elif use_optimizer_backward:  # colossalai style
            optimizer.backward(loss)
            if clip_grad_norm > 0:
                optimizer.clip_grad_norm(model, clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()

        elif scaler is not None:  # torch & fairscale amp style
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if clip_grad_norm > 0:
                if use_integraded_clip_grad:  # fairscale style
                    model.clip_grad_norm_(clip_grad_norm)
                else:  # torch style
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

        else:  # torch & fairscale normal style
            loss.backward()
            if clip_grad_norm > 0:
                if use_integraded_clip_grad:  # fairscale style
                    model.clip_grad_norm_(clip_grad_norm)
                else:  # torch style
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            lr_scheduler.step()

        bwd_end = time.time()

        num_steps += 1
        num_tokens += batch_tokens

        fwd_time = fwd_end - fwd_start
        bwd_time = bwd_end - bwd_start
        used_time += fwd_time + bwd_time

        if rank == 0:
            progress.set_postfix(loss=loss.item(),
                                 lr=lr_scheduler.get_last_lr()[0],
                                 time_forward=fwd_time,
                                 time_backward=bwd_time,
                                 throughput=batch_tokens * world_size / (fwd_time + bwd_time + 1e-12))

    peak_mem = None
    if mem_monitor is not None:
        peak_mem = max(mem_monitor.finish())

    all_reduce(train_loss)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Train]: Loss = {train_loss.item() / (world_size * num_steps):.3f}'
    msg += f' | Throughput = {num_tokens.item() / (used_time + 1e-12):.3f} tokens/sec'
    if peak_mem is not None:
        msg += f' | Peak memory = {peak_mem / 1024:.3f} GB.'
    _print_log(msg)


def _test(epoch, rank, world_size, test_dataloader, model, criterion, mem_monitor, config):
    use_autocast = config['method'] == 'torch'

    model.eval()

    progress = range(len(test_dataloader))
    if rank == 0:
        progress = tqdm(progress, desc=f"[Epoch {epoch} / Test]")

    test_loss = torch.zeros(()).to(torch.float).to(rank)
    used_time = 0.
    num_steps = 0
    num_tokens = torch.zeros(()).to(torch.int).to(rank)
    data_iter = iter(test_dataloader)

    if mem_monitor is not None:
        mem_monitor.start()

    with torch.no_grad():
        for _ in progress:
            batch_start = time.time()

            batch = next(data_iter)
            input_ids = batch['input_ids'].to(rank).detach()
            attention_mask = batch['attention_mask'].to(rank).detach()
            labels = batch['labels'].to(rank).detach()
            batch_tokens = labels.numel()

            if use_autocast:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(outputs, labels)
            test_loss += loss

            batch_end = time.time()

            num_steps += 1
            num_tokens += batch_tokens

            batch_time = batch_end - batch_start
            used_time += batch_time

            if rank == 0:
                progress.set_postfix(loss=loss.item(),
                                     step_time=batch_time,
                                     perplexity=math.exp(loss.item()),
                                     throughput=batch_tokens * world_size / (batch_time + 1e-12))

    peak_mem = None
    if mem_monitor is not None:
        peak_mem = max(mem_monitor.finish())

    all_reduce(test_loss)
    reduced_loss = test_loss.item() / (world_size * num_steps)
    all_reduce(num_tokens)

    msg = f'[Epoch {epoch} / Test]: Loss = {reduced_loss:.3f}'
    msg += f' | Perplexity = {math.exp(reduced_loss):.3f}'
    msg += f' | Throughput = {num_tokens.item() / (used_time + 1e-12):.3f} tokens/sec'
    if peak_mem is not None:
        msg += f' | Peak memory = {peak_mem / 1024:.3f} GB.'
    _print_log(msg)


def train(model, train_data, test_data, criterion, optimizer, scaler, lr_scheduler, config):
    rank = get_rank()
    world_size = get_world_size()

    _print_log('Benchmark start.')

    mem_monitor = None
    if 'use_mem_monitor' in config and config['use_mem_monitor']:
        log_path = os.environ['LOG'] if 'LOG' in os.environ else '.'
        mem_monitor = AsyncMemoryMonitor(rank, path=log_path)

    for epoch in range(NUM_EPOCHS):
        _train(epoch, rank, world_size, train_data, model, criterion, optimizer, lr_scheduler, scaler, mem_monitor,
               config)
        _test(epoch, rank, world_size, test_data, model, criterion, mem_monitor, config)

    _print_log('Benchmark complete.')
