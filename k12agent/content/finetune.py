"""LoRA supervised fine-tuning of LLaMA-2-13B on a mathematics teaching corpus.

Data format (JSONL, one example per line)::

    {"question": "勾股定理是怎么推导出来的？", "stage": "junior", "difficulty": "on_level",
     "language": "zh", "output": { ...LessonContent JSON (see prompts.OUTPUT_SCHEMA)... }}

``--bootstrap-from-kg`` creates seed examples from the knowledge graph (one per
node x stage-appropriate difficulty) so that the model learns the output
contract; add your own teacher-written examples on top of it.

Usage::

    python -m k12agent.content.finetune --train data/sft_train.jsonl \
        --model meta-llama/Llama-2-13b-hf --output checkpoints/llama13b-math-lora
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from ..config import REPO_ROOT
from ..schemas import Difficulty, GenerationSettings, Stage
from .generator import TemplateContentGenerator
from .knowledge_base import KnowledgeBase
from .prompts import build_prompt, wrap_chat


def lesson_to_target(lesson) -> str:
    d = asdict(lesson)
    out = {
        "topic": d["topic"],
        "concepts": d["concepts"],
        "sections": [{
            "kind": s["kind"].value if hasattr(s["kind"], "value") else s["kind"],
            "title": s["title"], "bullets": s["bullets"],
            "formulas": [f["latex"] for f in s["formulas"]],
            "narration": s["narration"], "emphasis": s["emphasis"],
        } for s in d["sections"]],
        "extension": d["extension"],
    }
    return json.dumps(out, ensure_ascii=False)


def bootstrap_from_kg(kb: KnowledgeBase, languages=("zh", "en")) -> List[Dict]:
    gen = TemplateContentGenerator(kb)
    rows = []
    for node in kb.nodes.values():
        for lang in languages:
            topic = node.field("topic", lang)
            questions = [f"{topic}是什么？请讲解推导过程。", f"请举例说明{topic}的应用。"] if lang == "zh" else \
                        [f"What is {topic.lower()}? Please explain the derivation.", f"Give applications of {topic.lower()}."]
            for diff in Difficulty:
                settings = GenerationSettings(stage=node.stage, difficulty=diff, language=lang)
                lesson = gen.generate(questions[0], settings)
                for q in questions:
                    rows.append({"question": q, "stage": node.stage.value, "difficulty": diff.value,
                                 "language": lang, "node_id": node.id, "output": json.loads(lesson_to_target(lesson))})
    return rows


def to_text_pairs(rows: List[Dict], kb: KnowledgeBase) -> List[Dict[str, str]]:
    pairs = []
    for r in rows:
        settings = GenerationSettings(stage=Stage(r["stage"]), difficulty=Difficulty(r.get("difficulty", "on_level")),
                                      language=r.get("language", "zh"))
        node = kb.get(r["node_id"]) if r.get("node_id") else None
        nodes = [node] if node else kb.retrieve(r["question"], settings.stage)
        prompt = wrap_chat(build_prompt(r["question"], kb.context_block(nodes, settings.language), settings))
        target = r["output"] if isinstance(r["output"], str) else json.dumps(r["output"], ensure_ascii=False)
        pairs.append({"prompt": prompt, "target": " " + target + " </s>"})
    return pairs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", type=Path, help="JSONL training file")
    p.add_argument("--bootstrap-from-kg", action="store_true")
    p.add_argument("--kb", type=Path, default=REPO_ROOT / "k12agent/content/data/math_kg.json")
    p.add_argument("--model", default="meta-llama/Llama-2-13b-hf")
    p.add_argument("--output", type=Path, default=Path("checkpoints/llama13b-math-lora"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--dump-only", action="store_true", help="write the SFT JSONL and exit")
    args = p.parse_args(argv)

    kb = KnowledgeBase.load(args.kb)
    rows: List[Dict] = []
    if args.bootstrap_from_kg:
        rows += bootstrap_from_kg(kb)
    if args.train:
        with open(args.train, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        p.error("no training data: pass --train and/or --bootstrap-from-kg")
    pairs = to_text_pairs(rows, kb)
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "sft_pairs.jsonl", "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"{len(pairs)} SFT pairs written to {args.output / 'sft_pairs.jsonl'}")
    if args.dump_only:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    def tokenize(ex):
        p_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        t_ids = tok(ex["target"], add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[: args.max_len]
        labels = ([-100] * len(p_ids) + t_ids)[: args.max_len]  # loss only on the answer
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

    ds = Dataset.from_list(pairs).map(tokenize, remove_columns=["prompt", "target"])

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
        }

    trainer = Trainer(
        model=model, train_dataset=ds, data_collator=collate,
        args=TrainingArguments(
            output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            lr_scheduler_type="cosine", warmup_ratio=0.03, logging_steps=10, save_strategy="epoch",
            bf16=torch.cuda.is_available(), report_to=[]))
    trainer.train()
    model.save_pretrained(str(args.output))
    tok.save_pretrained(str(args.output))
    print(f"LoRA adapter saved to {args.output}; set content.lora_adapter in your config.")


if __name__ == "__main__":
    main()
