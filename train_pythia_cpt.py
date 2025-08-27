#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,math,argparse,shutil,random,torch,torch.nn.functional as F
from typing import List
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,DataCollatorForLanguageModeling,set_seed,TrainerCallback

def hf_hint():
    if "HF_ENDPOINT" not in os.environ:
        print("[提示] 未检测到 HF_ENDPOINT。可设置镜像：export HF_ENDPOINT=https://hf-mirror.com")

def prepare_tokenizer(model_id: str, seq_len: int):
    tok=AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.model_max_length=seq_len; tok.padding_side="right"; return tok

def chunk_texts(examples, tokenizer, block_size):
    text=(tokenizer.sep_token or "\n\n").join(examples["text"]) if "text" in examples else "\n\n".join([str(x) for x in examples.values()][0])
    ids=tokenizer(text, return_attention_mask=False, add_special_tokens=False)["input_ids"]
    total=(len(ids)//block_size)*block_size; ids=ids[:total]
    out={"input_ids":[ids[i:i+block_size] for i in range(0,total,block_size)]}; out["labels"]=list(out["input_ids"]); return out

@torch.no_grad()
def score_choices_ll(model, tokenizer, contexts: List[str], choices: List[List[str]], max_length: int, device, batch_size: int=2)->List[int]:
    preds=[]; model.eval()
    for i in range(0,len(contexts),1):
        ctx=contexts[i]; ends=choices[i]; scores=[]
        for e in ends:
            text=(ctx+" "+e).strip()
            enc=tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)
            input_ids=enc["input_ids"].to(device)
            ctx_ids=tokenizer(ctx, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"].to(device)
            ctx_len=min(ctx_ids.size(1), input_ids.size(1))
            out=model(input_ids=input_ids); logits=out.logits[:,:-1,:]; labels=input_ids[:,1:]
            end_len=max(1,input_ids.size(1)-ctx_len)
            logits_end=logits[:,-end_len:,:]; labels_end=labels[:,-end_len:]
            logprobs=F.log_softmax(logits_end,dim=-1); tok_lp=logprobs.gather(-1,labels_end.unsqueeze(-1)).squeeze(-1)
            score=tok_lp.sum().item()/(end_len**0.5); scores.append(score)
        preds.append(int(torch.tensor(scores).argmax().item()))
    return preds

def accuracy(labels: List[int], preds: List[int])->float:
    return sum(int(a==b) for a,b in zip(labels,preds))/max(1,len(labels))

class EvalAndCompressCallback(TrainerCallback):
    def __init__(self, tokenizer, args, device, hs_dataset=None, piqa_dataset=None, hs_samples=500, piqa_samples=500, max_length=1024, work_dir="outputs"):
        self.tk=tokenizer; self.args=args; self.device=device; self.hs=hs_dataset; self.piqa=piqa_dataset
        self.hs_n=hs_samples; self.piqa_n=piqa_samples; self.max_length=max_length; self.work_dir=work_dir; os.makedirs(self.work_dir,exist_ok=True)
    def _sample_dataset(self, ds, n):
        if ds is None: return None
        n=min(n,len(ds)); return ds.select(random.sample(range(len(ds)),n))
    def _eval_hellaswag(self, model):
        if self.hs is None: return None
        ds=self._sample_dataset(self.hs,self.hs_n); contexts,choices,labels=[],[],[]
        for ex in ds:
            ctx=ex.get("ctx",None) or ex.get("context","")
            ends=ex.get("endings",None) or ex.get("endings_list",None)
            if ends is None: ends=[ex.get(f"ending{i}","") for i in range(4)]
            contexts.append(ctx); choices.append(list(ends)); labels.append(int(ex["label"]))
        preds=score_choices_ll(model,self.tk,contexts,choices,self.max_length,self.device); return accuracy(labels,preds)
    def _eval_piqa(self, model):
        if self.piqa is None: return None
        ds=self._sample_dataset(self.piqa,self.piqa_n); contexts,choices,labels=[],[],[]
        for ex in ds:
            ctx=ex.get("goal",""); ends=[ex.get("sol1",""),ex.get("sol2","")]
            contexts.append(ctx); choices.append(ends); labels.append(int(ex["label"]))
        preds=score_choices_ll(model,self.tk,contexts,choices,self.max_length,self.device); return accuracy(labels,preds)
    def _compress_stub(self, ckpt_dir):
        if not ckpt_dir or not os.path.isdir(ckpt_dir): return None
        zip_path=os.path.join(self.work_dir,os.path.basename(ckpt_dir.rstrip("/"))+".zip")
        if os.path.exists(zip_path):
            try: os.remove(zip_path)
            except: pass
        print(f"[占位压缩] 打包 {ckpt_dir} -> {zip_path}"); shutil.make_archive(zip_path.replace(".zip",""),'zip',ckpt_dir); return zip_path
    def on_evaluate(self, args, state, control, **kwargs):
        trainer=kwargs.get("trainer"); model=trainer.model; step=int(state.global_step); log={}
        try:
            hs=self._eval_hellaswag(model)
            if hs is not None: log["eval_hellaswag_acc"]=hs
        except Exception as e: print(f"[警告] HellaSwag 评测失败：{e}")
        try:
            pq=self._eval_piqa(model)
            if pq is not None: log["eval_piqa_acc"]=pq
        except Exception as e: print(f"[警告] PIQA 评测失败：{e}")
        if log: trainer.log(log)
        if args.save_steps and step%args.save_steps==0 and step>0:
            ckpt_dir=os.path.join(args.output_dir,f"checkpoint-{step}")
            try:
                z=self._compress_stub(ckpt_dir)
                if z: trainer.log({"compressed_checkpoint_zip":z})
            except Exception as e: print(f"[警告] 占位压缩失败：{e}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model_id",type=str,default="EleutherAI/pythia-1.4b")
    p.add_argument("--output_dir",type=str,default="outputs/pythia14b-cpt")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--seq_len",type=int,default=1024)
    p.add_argument("--per_device_bs",type=int,default=1)
    p.add_argument("--grad_accum",type=int,default=16)
    p.add_argument("--max_steps",type=int,default=3000)
    p.add_argument("--eval_steps",type=int,default=1000)
    p.add_argument("--save_steps",type=int,default=1000)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--warmup_ratio",type=float,default=0.06)
    p.add_argument("--cosine",action="store_true")
    p.add_argument("--bf16_if_possible",action="store_true")
    p.add_argument("--grad_ckpt",action="store_true")
    p.add_argument("--hellaswag_eval_samples",type=int,default=500)
    p.add_argument("--piqa_eval_samples",type=int,default=500)
    p.add_argument("--num_proc",type=int,default=4)
    p.add_argument("--smoke_test",action="store_true")
    p.add_argument("--disable_amp",action="store_true")
    p.add_argument("--train_corpus",type=str,default="c4-small",choices=["c4-small","owt10k","auto","wikitext","owt"])
    args=p.parse_args()

    hf_hint(); set_seed(args.seed); torch.backends.cuda.matmul.allow_tf32=True
    device="cuda" if torch.cuda.is_available() else "cpu"
    is_sm8x=(device=="cuda" and torch.cuda.get_device_capability(0)[0]>=8)
    use_bf16=bool(args.bf16_if_possible and is_sm8x)
    use_fp16=bool((not use_bf16) and device=="cuda" and (not args.disable_amp))

    print("[信息] 加载 tokenizer 与模型...")
    tok=prepare_tokenizer(args.model_id,args.seq_len)
    model=AutoModelForCausalLM.from_pretrained(args.model_id,low_cpu_mem_usage=True,trust_remote_code=True)
    model.resize_token_embeddings(len(tok))
    if args.grad_ckpt:
        try: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        except TypeError: model.gradient_checkpointing_enable()
        model.config.use_cache=False
    model.to(device)

    print("[信息] 选择训练数据...")
    train_raw=None
    def try_load(tag):
        if tag=="owt": return load_dataset("Skylion007/openwebtext",split="train")
        if tag=="c4-small": return load_dataset("allenai/c4","en",split="train[:1%]")
        if tag=="owt10k": return load_dataset("stas/openwebtext-10k",split="train")
        if tag=="wikitext": return load_dataset("wikitext","wikitext-103-raw-v1",split="train")
        raise ValueError(tag)
    order=[]
    if args.train_corpus=="auto": order=["owt","c4-small","wikitext"]
    elif args.train_corpus in ["c4-small","owt10k","wikitext","owt"]: order=[args.train_corpus]
    else: order=["c4-small"]
    for key in order:
        try:
            print(f"[信息] 尝试加载 {key} ..."); train_raw=try_load(key); print(f"[信息] 已加载 {key}"); break
        except Exception as e:
            print(f"[警告] 加载 {key} 失败：{e}")
    if train_raw is None:
        print("[信息] 兜底：使用 WikiText-103 训练集"); train_raw=try_load("wikitext")

    eval_wt=load_dataset("wikitext","wikitext-103-raw-v1",split="validation")
    try: hellaswag=load_dataset("hellaswag",split="validation")
    except Exception as e: print(f"[警告] HellaSwag 加载失败，将跳过：{e}"); hellaswag=None
    try: piqa=load_dataset("piqa",split="validation")
    except Exception as e: print(f"[警告] PIQA 加载失败，将跳过：{e}"); piqa=None

    print("[信息] 正在对训练/评估数据做分块tokenize...")
    tokenized_train=train_raw.map(lambda ex:chunk_texts(ex,tok,args.seq_len),batched=True,remove_columns=train_raw.column_names,num_proc=max(1,args.num_proc),desc="Tokenizing train")
    tokenized_eval=eval_wt.map(lambda ex:chunk_texts(ex,tok,args.seq_len),batched=True,remove_columns=eval_wt.column_names,num_proc=max(1,args.num_proc),desc="Tokenizing eval")
    collator=DataCollatorForLanguageModeling(tokenizer=tok,mlm=False)

    total_steps=300 if args.smoke_test else args.max_steps
    logging_steps=max(10,args.eval_steps//10)
    training_args=TrainingArguments(
        output_dir=args.output_dir,overwrite_output_dir=True,do_train=True,do_eval=True,
        eval_strategy="steps",eval_steps=args.eval_steps,save_steps=args.save_steps,
        logging_steps=logging_steps,logging_first_step=True,save_total_limit=2,
        learning_rate=args.lr,weight_decay=0.1,max_steps=total_steps,
        per_device_train_batch_size=args.per_device_bs,per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine" if args.cosine else "linear",
        bf16=use_bf16,fp16=use_fp16,half_precision_backend="cuda_amp",
        gradient_checkpointing=args.grad_ckpt,dataloader_num_workers=2,report_to=[],remove_unused_columns=False,
    )

    trainer=Trainer(model=model,args=training_args,train_dataset=tokenized_train,eval_dataset=tokenized_eval,data_collator=collator,tokenizer=tok)
    cb=EvalAndCompressCallback(tokenizer=tok,args=training_args,device=device,hs_dataset=hellaswag,piqa_dataset=piqa,
                               hs_samples=args.hellaswag_eval_samples,piqa_samples=args.piqa_eval_samples,max_length=args.seq_len,
                               work_dir=os.path.join(args.output_dir,"compressed"))
    trainer.add_callback(cb)

    print("[信息] 开始训练..."); trainer.train(); print("[信息] 训练完成。")
    eval_out=trainer.evaluate(); eval_loss=eval_out.get("eval_loss",None)
    if eval_loss is not None:
        ppl=math.exp(min(20,eval_loss)); print(f"[结果] WikiText-103 验证 PPL: {ppl:.3f}")
    try:
        hs_final=cb._eval_hellaswag(model)
        if hs_final is not None: print(f"[结果] HellaSwag (subset) acc: {hs_final:.4f}")
    except Exception as e: print(f"[警告] 结束评 HellaSwag 失败：{e}")
    try:
        piqa_final=cb._eval_piqa(model)
        if piqa_final is not None: print(f"[结果] PIQA (subset) acc: {piqa_final:.4f}")
    except Exception as e: print(f"[警告] 结束评 PIQA 失败：{e}")
    trainer.save_model(); print(f"[完成] 已保存到：{args.output_dir}")

if __name__=="__main__": main()
