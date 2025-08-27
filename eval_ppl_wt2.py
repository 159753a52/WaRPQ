#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, math, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--hf_dataset", type=str, default="wikitext")
    p.add_argument("--hf_config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--max_length", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hf_endpoint", type=str, default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    args = p.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
# 评测与训练对齐：不加 special tokens
    enc = tokenizer("\n\n".join(load_dataset(args.hf_dataset, args.hf_config, split=args.split)["text"]),
                    return_tensors="pt", add_special_tokens=False)

    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager").to(args.device)
    model.eval()

    input_ids = enc.input_ids.to(args.device)
    max_len = args.max_length or getattr(model.config, "n_positions", None) \
            or getattr(model.config, "max_position_embeddings", 1024)
    stride = args.stride
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    ntokens = 0

    with torch.no_grad():
        for i in range(0, seq_len, stride):
            begin_loc = max(i + stride - max_len, 0)
            end_loc   = min(i + stride, seq_len)
            trg_len   = end_loc - i            # 仅新 token 数
            if trg_len <= 0:
                continue

            inp = input_ids[:, begin_loc:end_loc]
            tgt = inp.clone()
            tgt[:, :-trg_len] = -100           # 只对最后 trg_len 个位置计损失

            out = model(inp, labels=tgt)
            # HF 的 loss 已按有效标签求平均，这里乘以参与平均的 token 数
            nll_sum += float(out.loss) * trg_len
            ntokens += trg_len

    avg_nll = nll_sum / max(ntokens, 1)
    ppl = math.exp(min(avg_nll, 80.0))
    print(f"PPL({args.split})={ppl:.4f} | avgNLL={avg_nll:.6f} | tokens={ntokens} | stride={stride} | max_len={max_len}")

if __name__ == "__main__":
    main()
