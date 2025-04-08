"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import csv
import json
from collections import namedtuple
from enum import Enum
from typing import List

import torch
from tg4perfetto import TraceGenerator


class EventType(Enum):
    kBegin = 0
    kEnd = 1
    kInstant = 2


def decode_tag(tag, num_blocks, num_groups):
    block_group_tag = tag >> 12
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (
        block_group_tag // num_groups,
        block_group_tag % num_groups,
        event_idx,
        event_type,
    )

def decode_pod_tag(tag):
    sm_idx_tag = tag >> 24
    block_idx_tag = (tag >> 12) & 0xFFF
    event_idx = (tag >> 2) & 0x3FF
    event_type = tag & 0x3
    return (
        sm_idx_tag, # sm_idx_tag
        block_idx_tag, # group_idx
        event_idx,
        event_type,
    )


def export_to_perfetto_trace(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_groups = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_groups = int(num_groups)

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for block_idx in range(num_blocks):
        pid = tgen.create_group(f"block_{block_idx}")
        for group_idx in range(num_groups):
            tid = pid.create_group(f"group_{group_idx}")
            tid_map[(block_idx, group_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        block_idx, group_idx, event_idx, event_type = decode_tag(
            tag, num_blocks, num_groups
        )
        event = event_names[event_idx]
        tid = tid_map[(block_idx, group_idx)]

        if (block_idx, group_idx, event_idx) in track_map:
            track = track_map[(block_idx, group_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(block_idx, group_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()


def export_to_perfetto_trace_pod(
    profiler_buffer: torch.Tensor,
    event_names: List[str],
    file_name: str,
) -> None:

    profiler_buffer_host = profiler_buffer.cpu()
    num_blocks, num_sms = profiler_buffer_host[:1].view(dtype=torch.int32)
    num_decode_blocks, num_prefill_blocks = profiler_buffer_host[1:2].view(dtype=torch.int32)
    num_blocks = int(num_blocks)
    num_sms = int(num_sms)
    num_decode_blocks = int(num_decode_blocks)
    num_prefill_blocks = int(num_prefill_blocks)
    num_groups = 1

    print(f"num_blocks: {num_blocks}, num_sms: {num_sms}, num_decode_blocks: {num_decode_blocks}, num_prefill_blocks: {num_prefill_blocks}")

    tgen = TraceGenerator(file_name)

    tid_map = {}
    track_map = {}
    for sm_idx in range(num_sms):
        pid = tgen.create_group(f"sm_{sm_idx}")
        for block_idx in range(num_blocks):
            tid = pid.create_group(f"block_{block_idx}")
            tid_map[(sm_idx, block_idx)] = tid

    for i in range(1, len(profiler_buffer_host)):
        if profiler_buffer_host[i] == 0:
            continue
        tag, timestamp = profiler_buffer_host[i : i + 1].view(dtype=torch.uint32)
        tag = int(tag)
        timestamp = int(timestamp)
        sm_idx, block_idx, event_idx, event_type = decode_pod_tag(tag)
        event = event_names[event_idx]
        tid = tid_map[(sm_idx, block_idx)]

        if (sm_idx, block_idx, event_idx) in track_map:
            track = track_map[(sm_idx, block_idx, event_idx)]
        else:
            track = tid.create_track()
            track_map[(sm_idx, block_idx, event_idx)] = track

        if event_type == EventType.kBegin.value:
            track.open(timestamp, event)
        elif event_type == EventType.kEnd.value:
            track.close(timestamp)
        elif event_type == EventType.kInstant.value:
            track.instant(timestamp, event)

    tgen.flush()