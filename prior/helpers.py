import torch
import numpy as np
from functools import partial, wraps

from math import ceil
from collections.abc import Iterable


def noop(*args, **kwargs):
    pass

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def split_args_and_kwargs(*args, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

def prior_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]
        return torch.cat(outputs, dim = 0)
    return inner