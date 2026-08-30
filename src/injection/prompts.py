from __future__ import annotations

CONDITIONS = ["C0", "C1", "C2", "C3", "C4"]


def options_block(options: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def build_prompt(question: str, options: dict[str, str], condition: str,
                 prompts_cfg: dict, evidence_text: str | None = None,
                 position: str = "before_question") -> str:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")

    instruction = (prompts_cfg["cot_instruction"] if condition in ("C1", "C3")
                   else prompts_cfg["answer_instruction"])
    # official ShotBench format: Question/Options headers, instruction last
    qblock = (f"Question: {question}\nOptions:\n{options_block(options)}\n"
              f"{instruction}")

    if condition in ("C2", "C3"):
        if not evidence_text:
            raise ValueError(f"condition {condition} requires evidence_text")
        if position == "before_question":
            return f"{evidence_text}\n\n{qblock}"
        return f"{qblock}\n\n{evidence_text}"
    return qblock
