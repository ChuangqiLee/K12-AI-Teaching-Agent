"""Stage- and band-specific prompt templates (Section 2.1, Section 4.1.1).

The output contract is a JSON object following the instructional sequence
"definition -> derivation -> application", which is then mapped onto slides.
NOTE: the original templates are listed in Tables A2-A4 of the online
appendix; the versions below are re-written from the descriptions in the paper.
"""

from __future__ import annotations

from ..schemas import Difficulty, GenerationSettings, Stage

OUTPUT_SCHEMA = """{
  "topic": str,
  "concepts": [str],
  "sections": [
    {"kind": "definition" | "derivation" | "application" | "exercise" | "summary",
     "title": str,
     "bullets": [str],            # on-slide text, <= 3 short lines each
     "formulas": [str],           # LaTeX without $ delimiters
     "narration": str,            # what the virtual teacher says for this slide
     "emphasis": [str]}           # key words to stress in speech
  ],
  "extension": str | null
}"""

STAGE_STYLE = {
    Stage.PRIMARY: {
        "zh": "你是一位亲切耐心的小学数学老师，面对五年级学生。使用简单、口语化的语言，多用生活中的例子（蛋糕、披萨、水果），每页不超过三句话，避免专业术语，必要时用“一步一步”的动画式讲解。",
        "en": "You are a warm, patient primary-school maths teacher for Grade 5. Use simple spoken language and everyday examples (cakes, pizza, fruit). At most three sentences per slide, avoid jargon, explain step by step.",
    },
    Stage.JUNIOR: {
        "zh": "你是一位经验丰富的初中数学老师，面对八年级学生。先给出准确定义，再给出完整的推导或证明，最后结合建筑、导航等真实情境举例。",
        "en": "You are an experienced junior-high maths teacher for Grade 8. Give a precise definition, then a complete derivation or proof, then real-world applications such as architecture or navigation.",
    },
    Stage.SENIOR: {
        "zh": "你是一位严谨的高中数学老师，面对十一年级学生。必须给出完整的逐步推导（每一步只做一个变形并说明理由），不得跳步；明确写出前提条件与结论之间的逻辑关系；给出与高考难度相当的例题。",
        "en": "You are a rigorous senior-high maths teacher for Grade 11. Always give a complete step-by-step derivation (one transformation per step, with justification) and never skip steps; state the logical relation between premises and conclusion; include exam-level worked examples.",
    },
}

DIFFICULTY_HINT = {
    Difficulty.REMEDIAL: {
        "zh": "该学生基础薄弱：先复习前置知识（见知识库中的 prerequisite），使用带逐步淡出的例题，并在每一步后加入提示。",
        "en": "The learner is below level: review prerequisites first, use worked examples with fading and add a hint after each step.",
    },
    Difficulty.ON_LEVEL: {
        "zh": "该学生处于年级平均水平：按标准难度讲解，并安排一道巩固练习。",
        "en": "The learner is on level: teach at standard difficulty and add one consolidation exercise.",
    },
    Difficulty.ENRICHMENT: {
        "zh": "该学生基础扎实：减少脚手架，增加拓展内容（如推广到三维空间）和一道挑战题。",
        "en": "The learner is proficient: reduce scaffolds, add extension content (e.g. generalisation to 3-D) and one challenge problem.",
    },
}

TEMPLATE_ZH = """{style}
{difficulty}
{scaffolds}
{local}
请严格依据下面的知识库内容回答学生的问题，不要编造知识库以外的结论。
### 知识库
{knowledge}

### 学生问题
{question}

### 输出要求
只输出一个 JSON 对象（不要输出其他文字），结构如下：
{schema}
sections 必须按照 definition → derivation → application 的顺序，每个 section 对应一页幻灯片；
narration 是老师口头讲解的完整句子，数学式用中文读法写出（例如 “a 的平方加 b 的平方等于 c 的平方”）。
"""

TEMPLATE_EN = """{style}
{difficulty}
{scaffolds}
{local}
Answer strictly based on the knowledge base below; do not invent results outside it.
### Knowledge base
{knowledge}

### Student question
{question}

### Output format
Output only one JSON object (no other text) with this structure:
{schema}
Sections must follow the order definition -> derivation -> application; one section per slide.
"narration" contains the full spoken explanation, with formulas verbalised (e.g. "a squared plus b squared equals c squared").
"""


def scaffold_text(settings: GenerationSettings) -> str:
    s = settings.scaffolds
    zh = settings.language == "zh"
    parts = []
    if s.worked_examples:
        parts.append("包含一道完整的例题" if zh else "include a fully worked example")
    if s.stepwise_derivation:
        parts.append("推导必须逐步展开" if zh else "derivations must be stepwise")
    if s.visual_signaling:
        parts.append("在 emphasis 中标出关键词以便高亮" if zh else "list key words in 'emphasis' for visual signalling")
    if s.targeted_hints:
        parts.append("每一步后给出提示" if zh else "add a hint after each step")
    if s.fading:
        parts.append("例题的提示逐步减少" if zh else "fade the hints across examples")
    if settings.pause_prompts:
        parts.append("在关键处加入“想一想”停顿提示" if zh else "insert 'think about it' pause prompts at key points")
    sep = "；" if zh else "; "
    return ("脚手架：" if zh else "Scaffolds: ") + sep.join(parts) if parts else ""


def build_prompt(question: str, knowledge: str, settings: GenerationSettings) -> str:
    lang = "zh" if settings.language == "zh" else "en"
    local = ""
    if settings.localization:
        local += ("情境本地化：" if lang == "zh" else "Localise examples: ") + settings.localization + "\n"
    if settings.reading_level:
        local += ("阅读水平：" if lang == "zh" else "Reading level: ") + settings.reading_level + "\n"
    if settings.terminology:
        local += ("术语要求：" if lang == "zh" else "Terminology: ") + settings.terminology + "\n"
    if settings.subject != "math":
        from ..subjects.hooks import subject_prompt_addendum
        addendum = subject_prompt_addendum(settings.subject, lang)
        if addendum:
            local += addendum + "\n"
    template = TEMPLATE_ZH if lang == "zh" else TEMPLATE_EN
    return template.format(
        style=STAGE_STYLE[settings.stage][lang],
        difficulty=DIFFICULTY_HINT[settings.difficulty][lang],
        scaffolds=scaffold_text(settings),
        local=local.strip(),
        knowledge=knowledge or ("（无匹配条目）" if lang == "zh" else "(no matching entry)"),
        question=question,
        schema=OUTPUT_SCHEMA,
    )


def wrap_chat(prompt: str) -> str:
    """LLaMA-2 chat format (also fine for the base model after LoRA SFT)."""
    return f"<s>[INST] {prompt.strip()} [/INST]"
