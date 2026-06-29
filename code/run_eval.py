"""评测入口。

用法:
  python run_eval.py --model qwen3_omni --config ../configs/eval_config.yaml
  python run_eval.py --model qwen3_omni --conditions E0,E1,E2 --limit 6   # 冒烟
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # code/

import yaml  # noqa: E402
from common.runner import run_eval  # noqa: E402


def get_adapter(name, mcfg):
    if name == "qwen3_omni":
        from models.qwen3_omni import Qwen3OmniAdapter
        return Qwen3OmniAdapter(mcfg["path"])
    if name == "minicpm_o":
        from models.minicpm_o import MiniCPMoAdapter
        return MiniCPMoAdapter(mcfg["path"])
    if name == "ming":
        from models.ming import MingOmniAdapter
        return MingOmniAdapter(mcfg["path"], weights=mcfg.get("weights"))
    raise ValueError(f"unknown model {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs", "eval_config.yaml"))
    ap.add_argument("--conditions", default=None, help="逗号分隔, 覆盖配置")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条(冒烟)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    conditions = (args.conditions.split(",") if args.conditions
                  else cfg.get("conditions", ["E0", "E1", "E2"]))

    if args.limit > 0:
        # 冒烟：临时截断 subset
        import json
        sub = [json.loads(l) for l in open(cfg["subset_path"]) if l.strip()][:args.limit]
        tmp = cfg["subset_path"] + f".smoke{args.limit}"
        with open(tmp, "w") as f:
            for r in sub:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cfg["subset_path"] = tmp
        print(f"[smoke] using {len(sub)} samples -> {tmp}")

    adapter = get_adapter(args.model, cfg["models"][args.model])
    run_eval(adapter, cfg, conditions)


if __name__ == "__main__":
    main()
