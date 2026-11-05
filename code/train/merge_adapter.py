# Merge a LoRA adapter into the base model (CPU ok). Env: BASE_MODEL, ADAPTER (dir), OUT (merged dir)
import os, torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
base=os.environ.get("BASE_MODEL","Qwen/Qwen2.5-VL-7B-Instruct"); ad=os.environ["ADAPTER"]; out=os.environ["OUT"]
m=Qwen2_5_VLForConditionalGeneration.from_pretrained(base, torch_dtype=torch.bfloat16)
m=PeftModel.from_pretrained(m, ad).merge_and_unload(); m.save_pretrained(out, safe_serialization=True)
AutoProcessor.from_pretrained(base).save_pretrained(out); print("merged ->", out)
