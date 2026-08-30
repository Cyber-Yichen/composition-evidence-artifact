from __future__ import annotations

import re


def parse_choice(text: str, options: dict[str, str]) -> tuple[str | None, str]:
    """Returns (letter | None, method)."""
    if not text:
        return None, "empty"
    letters = "".join(sorted(options.keys()))
    text = text.strip()

    # 0. reasoning-format models: <answer>D</answer> (take the last tag)
    tags = re.findall(rf"<answer>\s*\(?([{letters}])\b[^<]*</answer>", text,
                      flags=re.IGNORECASE)
    if tags:
        return tags[-1], "answer_tag"

    # 1. response begins with the letter: "D", "D.", "(D)", "**D**", "D. Medium Wide"
    m = re.match(rf"^[\s\*\(\[]*([{letters}])[\s\*\)\]\.:,]", text + " ")
    if m:
        return m.group(1), "prefix"

    # 2. "answer is D" / "Answer: D" / "option (D)" — take the LAST occurrence
    #    (CoT responses conclude at the end)
    hits = re.findall(
        rf"(?:answer|option|choice)\s*(?:is|:)?\s*[\*\(\[]*([{letters}])\b",
        text, flags=re.IGNORECASE)
    if hits:
        return hits[-1], "keyword"

    # 3. last line begins with a letter (CoT "final answer on the last line")
    last = text.splitlines()[-1].strip()
    m = re.match(rf"^[\s\*\(\[]*([{letters}])[\s\*\)\]\.:,]*$", last)
    if m:
        return m.group(1), "lastline"

    # 4. exactly one option's text appears verbatim in the response
    matches = [k for k, v in options.items() if v.lower() in text.lower()]
    if len(matches) == 1:
        return matches[0], "text_match"

    # 5. a standalone letter anywhere, if unambiguous
    standalone = set(re.findall(rf"\b([{letters}])\b", text))
    if len(standalone) == 1:
        return standalone.pop(), "standalone"

    return None, "fail"
