"""Real teacher, isolated FSDP CPUOffload versus explicit parameter placement."""

import argparse
import functools
import json
import os
import time
from pathlib import Path

import psutil
import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # noqa: N817
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["cpuoffload", "cpuoffload-frozen", "explicit"], required=True
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29591", RANK="0", WORLD_SIZE="1")
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    process = psutil.Process()

    def record(stage, **extra):
        torch.cuda.synchronize()
        mem = process.memory_full_info()
        row = dict(
            stage=stage,
            time=time.time(),
            rss=mem.rss,
            pss=mem.pss,
            uss=mem.uss,
            available=psutil.virtual_memory().available,
            gpu_allocated=torch.cuda.memory_allocated(),
            gpu_reserved=torch.cuda.memory_reserved(),
            pinned=torch.cuda.memory.host_memory_stats(),
            **extra,
        )
        with (args.output / "stages.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    record("baseline")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    record(
        "hf_loaded", parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters())
    )
    cpuoffload = args.mode.startswith("cpuoffload")
    if args.mode != "cpuoffload":
        model.requires_grad_(False)
    model = FSDP(
        model,
        device_id=0,
        use_orig_params=False,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={LlamaDecoderLayer}
        ),
        cpu_offload=CPUOffload(offload_params=True) if cpuoffload else None,
    )
    record("fsdp_constructed")
    inputs = torch.full((1, 256), 42, device="cuda", dtype=torch.long)
    for iteration in range(3):
        record(f"forward_{iteration}_before")
        with torch.no_grad():
            output = model(input_ids=inputs, use_cache=False).logits
            result = output[:, -1].float().cpu()
        del output
        grad_bytes = sum(
            getattr(h.flat_param, "_cpu_grad", torch.empty(0)).numel()
            * getattr(h.flat_param, "_cpu_grad", torch.empty(0)).element_size()
            for h in model._all_handles  # noqa: SLF001 -- buffer ownership audit
        )
        record(f"forward_{iteration}_after", cpu_gradient_bytes=grad_bytes)
        if iteration == 0:
            buffers = []
            for handle in model._all_handles:  # noqa: SLF001 -- FSDP profiling
                for name in ("_local_shard", "_cpu_grad"):
                    tensor = getattr(handle.flat_param, name, None)
                    if tensor is not None:
                        size = tensor.untyped_storage().nbytes()
                        buffers.append(
                            {
                                "name": name,
                                "bytes": size,
                                "rounded_bytes": 1 << (size - 1).bit_length(),
                                "pinned": tensor.is_pinned(),
                                "device": str(tensor.device),
                            }
                        )
            (args.output / "buffers.json").write_text(json.dumps(buffers, indent=2) + "\n")
        if not cpuoffload:
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            offload_fsdp_model_to_cpu(model)
            record(f"explicit_offload_{iteration}")
            torch._C._host_emptyCache()  # noqa: SLF001 -- fixed PyTorch 2.8
            record(f"explicit_trim_{iteration}")
            load_fsdp_model_to_gpu(model)
        if iteration == 0:
            torch.save(result, args.output / "last-token-logits.pt")
        time.sleep(1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
