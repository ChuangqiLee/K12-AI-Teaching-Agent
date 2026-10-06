"""Content generation: question -> structured LessonContent (Section 2.1 / 3.1).

Two backends share one interface:

* ``LlamaContentGenerator`` - LLaMA-2-13B (+ optional LoRA adapter from
  :mod:`k12agent.content.finetune`) through HuggingFace ``transformers``.
* ``TemplateContentGenerator`` - deterministic, knowledge-graph-only generator
  used for offline demos, unit tests and machines without a GPU.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from ..config import resolve_path
from ..schemas import (ContentSection, Difficulty, Formula, GenerationSettings,
                       LessonContent, PipelineError, SectionKind)
from .knowledge_base import KGNode, KnowledgeBase
from .prompts import build_prompt, wrap_chat

log = logging.getLogger(__name__)


class ContentGenerator:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def generate(self, question: str, settings: GenerationSettings) -> LessonContent:
        raise NotImplementedError

    def _retrieve(self, question: str, settings: GenerationSettings) -> List[KGNode]:
        with_prereq = settings.difficulty == Difficulty.REMEDIAL
        return self.kb.retrieve(question, settings.stage, top_k=1, with_prerequisites=with_prereq)


# --------------------------------------------------------------------------- #
# JSON parsing (robust to chatty model output)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model output")


_KIND_ALIASES = {"proof": "derivation", "example": "application", "practice": "exercise"}


def lesson_from_json(obj: dict, question: str, settings: GenerationSettings,
                     raw: Optional[str] = None) -> LessonContent:
    sections = []
    for s in obj.get("sections", []):
        kind = str(s.get("kind", "definition")).lower()
        kind = _KIND_ALIASES.get(kind, kind)
        try:
            kind_enum = SectionKind(kind)
        except ValueError:
            kind_enum = SectionKind.SUMMARY
        bullets = [str(b) for b in s.get("bullets", []) if str(b).strip()]
        sections.append(ContentSection(
            kind=kind_enum,
            title=str(s.get("title", kind_enum.value)),
            bullets=bullets,
            narration=str(s.get("narration") or " ".join(bullets)),
            formulas=[Formula(latex=str(f).strip("$ ")) for f in s.get("formulas", [])],
            emphasis=[str(e) for e in s.get("emphasis", [])],
        ))
    if not sections:
        raise ValueError("model output contains no sections")
    return LessonContent(
        question=question,
        topic=str(obj.get("topic", question)),
        stage=settings.stage,
        concepts=[str(c) for c in obj.get("concepts", [])],
        sections=sections,
        extension=obj.get("extension"),
        raw_model_output=raw,
    )


# --------------------------------------------------------------------------- #
# LLaMA backend
# --------------------------------------------------------------------------- #
class LlamaContentGenerator(ContentGenerator):
    def __init__(self, kb: KnowledgeBase, model_name_or_path: str, lora_adapter: Optional[str] = None,
                 load_in_4bit: bool = True, max_new_tokens: int = 1024, temperature: float = 0.3,
                 top_p: float = 0.9, max_retries: int = 2):
        super().__init__(kb)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        kwargs = {"device_map": "auto"}
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        if lora_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(resolve_path(lora_adapter)))
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

    def _complete(self, prompt: str) -> str:
        inputs = self.tokenizer(wrap_chat(prompt), return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5), top_p=self.top_p,
                repetition_penalty=1.05, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, question: str, settings: GenerationSettings) -> LessonContent:
        nodes = self._retrieve(question, settings)
        prompt = build_prompt(question, self.kb.context_block(nodes, settings.language), settings)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            raw = self._complete(prompt)
            try:
                lesson = lesson_from_json(_extract_json(raw), question, settings, raw)
                if nodes and not lesson.extension and settings.difficulty == Difficulty.ENRICHMENT:
                    lesson.extension = nodes[0].field("extension", settings.language)
                return lesson
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err
                log.warning("LLaMA output not parseable (attempt %d): %s", attempt + 1, err)
        if nodes:  # graceful degradation: fall back to the knowledge graph
            log.warning("falling back to template generator")
            return TemplateContentGenerator(self.kb).generate(question, settings)
        raise PipelineError("content", f"could not parse model output: {last_err}")


# --------------------------------------------------------------------------- #
# Knowledge-graph template backend
# --------------------------------------------------------------------------- #
class TemplateContentGenerator(ContentGenerator):
    """Builds the definition -> derivation -> application deck directly from the KG."""

    def generate(self, question: str, settings: GenerationSettings) -> LessonContent:
        lang = settings.language
        zh = lang == "zh"
        nodes = self._retrieve(question, settings)
        if not nodes:
            raise PipelineError("content", "no knowledge-graph node matches the question; "
                                           "use the LLaMA backend or extend math_kg.json")
        main = nodes[0]
        chunk = max(1, settings.micro_chunk_sentences)
        sections: List[ContentSection] = []

        for pre in nodes[1:]:  # prerequisite review for the remedial band
            sections.append(ContentSection(
                kind=SectionKind.DEFINITION,
                title=("先复习：" if zh else "Review: ") + str(pre.field("topic", lang)),
                bullets=[pre.field("definition", lang)],
                narration=pre.field("definition", lang),
                formulas=[Formula(f) for f in pre.data.get("formulas", [])[:1]],
            ))

        definition = main.field("definition", lang)
        concepts = main.field("concepts", lang) or []
        sections.append(ContentSection(
            kind=SectionKind.DEFINITION,
            title=("定义：" if zh else "Definition: ") + str(main.field("topic", lang)),
            bullets=[definition],
            narration=definition,
            formulas=[Formula(main.data["formulas"][0])] if main.data.get("formulas") else [],
            emphasis=concepts[:3],
        ))

        steps = main.field("derivation", lang) or []
        if settings.scaffolds.stepwise_derivation:
            groups = [steps[i:i + chunk] for i in range(0, len(steps), chunk)]
        else:
            groups = [steps[-1:]] if steps else []
        for gi, group in enumerate(groups):
            narration = " ".join(group)
            if settings.pause_prompts and gi == 0:
                narration += " 想一想，下一步该怎么做？" if zh else " Think about it: what comes next?"
            sections.append(ContentSection(
                kind=SectionKind.DERIVATION,
                title=("推导" if zh else "Derivation") + (f" ({gi + 1}/{len(groups)})" if len(groups) > 1 else ""),
                bullets=list(group),
                narration=narration,
                formulas=[Formula(f) for f in main.data.get("formulas", [])[1:2]] if gi == len(groups) - 1 else [],
            ))

        apps = main.field("applications", lang) or []
        if apps:
            take = apps if settings.scaffolds.worked_examples else apps[:1]
            sections.append(ContentSection(
                kind=SectionKind.APPLICATION,
                title="应用举例" if zh else "Applications",
                bullets=list(take),
                narration=" ".join(take),
            ))

        extension = main.field("extension", lang)
        if settings.difficulty == Difficulty.ENRICHMENT and extension:
            sections.append(ContentSection(
                kind=SectionKind.SUMMARY,
                title="拓展" if zh else "Extension",
                bullets=[extension], narration=extension))

        return LessonContent(question=question, topic=str(main.field("topic", lang)),
                             stage=settings.stage, concepts=list(concepts),
                             sections=sections, extension=extension)


def build_generator(cfg) -> ContentGenerator:
    kb = KnowledgeBase.load(resolve_path(cfg.content.knowledge_base))
    if cfg.content.backend == "template":
        return TemplateContentGenerator(kb)
    if cfg.content.backend == "hf":
        c = cfg.content
        return LlamaContentGenerator(kb, c.model_name_or_path, c.lora_adapter, c.load_in_4bit,
                                     c.max_new_tokens, c.temperature, c.top_p)
    raise ValueError(f"unknown content backend {cfg.content.backend!r}")
