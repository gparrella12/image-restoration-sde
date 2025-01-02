import os
import torch
import torch.distributed as dist

def main():
    import os
    import torch
    import torch.distributed as dist

    print("=== ENVIRONMENT VARIABLES ===")
    for k, v in os.environ.items():
        if k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "CUDA_VISIBLE_DEVICES"):
            print(f"{k} = {repr(v)}")
    print("=============================")

    dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    current_device = torch.cuda.current_device()

    print(f"[INFO] Rank {rank} di {world_size} => uso GPU locale {current_device}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()