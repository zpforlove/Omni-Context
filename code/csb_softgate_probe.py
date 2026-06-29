"""P3.5 软门控探针：测哨兵 π(COMPLEX) 连续概率在 干净流 vs 复杂流 的分布。
核心问题：训练后硬门控 argmax 塌缩(干净流也输出COMPLEX)，但连续概率 π 是否仍保留区分度？
  有区分度 → 三档软门控(≥0.8全线索/0.5-0.8弱线索/<0.5不注入)能救塌缩，对标IRAF连续门控。
  无区分度 → π 也饱和，软门控救不了，必须靠 A(clean上采样+w_gate=3 续训)。
forced-choice：哨兵prompt后拼"GATE:"，单次解码取首token在 ' COM'(COMPLEX) vs ' CLEAN' 的softmax。
"""
import argparse, json, sys, statistics as st
sys.path.insert(0, "/cpfs_speech3/yulian.zpf/Omni-Context/code")
import torch, torch.nn.functional as Fn
import run_bench_eval as R
from gdpo_chain_train import build_prompt
from csb_eval_run import load_adapter, rows_of
ROOT = R.ROOT
CID, LID = 7682, 77000   # ' COM'(COMPLEX首token) vs ' CLEAN'


@torch.no_grad()
def pi_complex(ad, wav, verbose=False):
    conv = [{"role": "system", "content": [{"type": "text", "text": ad.sys_prompt}]},
            {"role": "user", "content": [{"type": "audio", "audio": wav}]
             + [{"type": "text", "text": build_prompt(None, None)}]}]
    text = ad.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) + "GATE:"
    audios, images, videos = ad._process_mm_info(conv, use_audio_in_video=False)
    inputs = ad.processor(text=text, audio=audios, images=images, videos=videos,
                          return_tensors="pt", padding=True, use_audio_in_video=False)
    inputs = inputs.to(ad.model.device).to(ad.model.dtype)
    # Qwen3-Omni 的 model.generate 返回 tuple 拿不到 scores；直接 thinker forward 取末位 logits
    lm = ad.model.thinker if hasattr(ad.model, "thinker") else ad.model
    out = lm(**inputs)
    sc = out.logits[0, -1, :].float()
    if verbose:
        top = int(sc.argmax()); print("  top1=", top, repr(ad.processor.tokenizer.decode([top])),
                                       "logit_COM=%.2f logit_CLEAN=%.2f" % (sc[CID], sc[LID]))
    p = Fn.softmax(torch.stack([sc[CID], sc[LID]]), dim=0)
    return float(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_omni")
    ap.add_argument("--lora", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    ad = load_adapter(a.model, a.lora or None)
    rows = [("mix", r) for r in rows_of("eval", "en", "mix", a.n)] \
         + [("clean", r) for r in rows_of("eval", "en", "clean", a.n)]
    out = []
    for kind, r in rows:
        try:
            pi = pi_complex(ad, r["wav"], a.verbose)
        except Exception as e:
            print("ERR", r["id"], repr(e)[:100]); continue
        out.append({"id": r["id"], "kind": kind, "pi": pi})
        if a.verbose:
            print(f"  {kind} {r['id']} pi={pi:.3f}")
    json.dump(out, open(ROOT + f"/results/softgate_pi__{a.tag}.json", "w"))
    print("=== π(COMPLEX) 分布 ===")
    for k in ["mix", "clean"]:
        ps = sorted(x["pi"] for x in out if x["kind"] == k)
        if ps:
            n = len(ps)
            print(f"{k}: n={n} mean={st.mean(ps):.3f} median={ps[n//2]:.3f} "
                  f"p10={ps[n//10]:.3f} p90={ps[min(n*9//10,n-1)]:.3f} "
                  f">=0.8={sum(p>=0.8 for p in ps)/n:.2f} 0.5-0.8={sum(0.5<=p<0.8 for p in ps)/n:.2f} <0.5={sum(p<0.5 for p in ps)/n:.2f}")
    print("DONE", a.tag)


if __name__ == "__main__":
    main()
