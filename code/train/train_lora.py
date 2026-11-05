#!/usr/bin/env python3
"""Portable LoRA trainer for Qwen2.5-VL-7B computer-use (no Modal). Runs on any single 48-80 GB GPU (H100/A100-80G/L40S/A6000*).
MODE=sft  : weighted NLL on assistant tokens (weight field optional, default 1).
MODE=grpo : rows carry signed weights from grpo_select.py (group-relative advantages; negatives = failures / false 'done').
            loss = w * NLL  (negative rows clipped: no push once NLL > NEG_NLL_CAP)  +  KL_BETA * KL(pi || pi_ref), pi_ref = INIT (adapter disabled).
Env vars (all optional except DATA):
  BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct  (processor/tokenizer source)    INIT=<path or HF id of weights to start from, e.g. ./ckpt/sft_v3/merged>
  DATA=<jsonl>  DATA_ROOT=<dir that image paths are relative to, default dirname(DATA)/..>   OUT=./ckpt/<RUN>   RUN=sft_v4
  MODE=sft|grpo  LR=1e-4  EPOCHS=1  GRAD_ACCUM=8  WARMUP=5  LORA_R=16  LORA_ALPHA=32  LORA_DROPOUT=0.05  MAX_LEN=4096  SEED=0
  KL_BETA=0.05 (grpo default)  NEG_NLL_CAP=1.5  SAVE_MERGED=1  HF_HOME (cache)
* 24 GB cards are not enough for bf16 7B + activations at 1260x784 screenshots."""
import json, os, random, time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
E = os.environ.get
mode = E("MODE", "sft"); run = E("RUN", "sft_v4"); data = E("DATA")
assert data, "set DATA=/path/to/train.jsonl"
cfg = dict(base=E("BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"), init=E("INIT", E("BASE_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")), data=data,
           root=E("DATA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(data)))), out=E("OUT", f"./ckpt/{run}"), run=run, mode=mode,
           lr=float(E("LR", "1e-4" if mode == "sft" else "2e-5")), epochs=int(E("EPOCHS", "1")), grad_accum=int(E("GRAD_ACCUM", "8")),
           warmup=int(E("WARMUP", "5")), r=int(E("LORA_R", "16")), alpha=int(E("LORA_ALPHA", "32")), dropout=float(E("LORA_DROPOUT", "0.05")),
           max_len=int(E("MAX_LEN", "4096")), seed=int(E("SEED", "0")), kl_beta=float(E("KL_BETA", "0" if mode == "sft" else "0.05")),
           neg_nll_cap=float(E("NEG_NLL_CAP", "1.5")), save_merged=E("SAVE_MERGED", "1") == "1",
           target=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
print(json.dumps(cfg, indent=1), flush=True)
random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
recs = [json.loads(l) for l in open(cfg["data"])]
proc = AutoProcessor.from_pretrained(cfg["base"])
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["init"], torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"], target_modules=cfg["target"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
def encode(r):
    img = Image.open(os.path.join(cfg["root"], r["image"])).convert("RGB")
    msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": r["user"]}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p = proc(text=[prompt], images=[img], return_tensors="pt"); f = proc(text=[prompt + r["target"] + "<|im_end|>\n"], images=[img], return_tensors="pt")
    lab = f["input_ids"].clone(); lab[:, :p["input_ids"].shape[1]] = -100; f["labels"] = lab
    return {k: v.cuda() for k, v in f.items()}
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=0.0)
total = max(1, cfg["epochs"] * len(recs) // cfg["grad_accum"]); sched = get_cosine_schedule_with_warmup(opt, cfg["warmup"], total)
log, stats, step, t0 = [], [], 0, time.time(); model.train()
os.makedirs(cfg["out"], exist_ok=True)
for ep in range(cfg["epochs"]):
    order = list(range(len(recs))); random.shuffle(order); acc = []
    for i, idx in enumerate(order):
        b = encode(recs[idx]); w = float(recs[idx].get("weight", 1.0))
        if cfg["kl_beta"] <= 0 and w >= 0:
            loss = model(**b).loss * w
        else:
            out = model(**b); nll = out.loss; m = b["labels"][:, 1:] != -100
            lp = torch.log_softmax(out.logits[:, :-1][m].float(), -1)
            if cfg["kl_beta"] > 0:
                with torch.no_grad(), model.disable_adapter(): rl = torch.log_softmax(model(**b).logits[:, :-1][m].float(), -1)
                kl = (lp.exp() * (lp - rl)).sum(-1).mean()
            else: kl = torch.zeros((), device=nll.device)
            pg = nll * w
            if w < 0 and nll.item() > cfg["neg_nll_cap"]: pg = pg * 0.0
            loss = pg + cfg["kl_beta"] * kl; stats.append({"w": w, "nll": round(nll.item(), 4), "kl": round(kl.item(), 5)})
        (loss / cfg["grad_accum"]).backward(); acc.append(loss.item())
        if (i + 1) % cfg["grad_accum"] == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
            e = {"step": step, "epoch": round(ep + (i + 1) / len(order), 4), "loss": sum(acc) / len(acc), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
            log.append(e); print(json.dumps(e), flush=True); acc = []
            json.dump({"cfg": cfg, "log": log, "partial": True}, open(f"{cfg['out']}/train_log.partial.json", "w"))
model.save_pretrained(f"{cfg['out']}/adapter")
if cfg["save_merged"]:
    merged = model.merge_and_unload(); merged.save_pretrained(f"{cfg['out']}/merged", safe_serialization=True); proc.save_pretrained(f"{cfg['out']}/merged")
json.dump({"cfg": cfg, "log": log, "n_samples": len(recs), "grpo_stats": stats}, open(f"{cfg['out']}/train_log.json", "w"), indent=1)
print("DONE", cfg["out"], flush=True)
