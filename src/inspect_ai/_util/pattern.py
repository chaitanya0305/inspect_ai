# Each pattern matches the LAST `ANSWER:` occurrence in the output (models
# sometimes self-correct, and CoT prompts instruct the answer on the last
# line). LETTER and WORD use a negative lookahead with DOTALL to skip earlier
# occurrences; LINE is anchored to end-of-string with \Z.
# LETTER does not match an answer that names several letters (`A/B`, `A & B`,
# `A or B`), so a hedged answer is not read as its first letter.
ANSWER_PATTERN_LETTER = (
    r"(?is)ANSWER\s*:\s*([A-Za-z])"
    r"(?![ \t]*(?:[/&;+|]|or\b)[ \t]*[A-Za-z]"
    r"(?=[ \t]*(?:[/&;+|]|or\b|[.!?]?[ \t]*(?:\r?\n|$))))"
    r"(?:[^\w]|\n|$)(?!.*ANSWER\s*:)"
)
ANSWER_PATTERN_WORD = (
    r"(?is)ANSWER\s*:\s*(\S+?)(?=[.,;:!?]?\s*(?:\n|$))(?!.*ANSWER\s*:)"
)
ANSWER_PATTERN_LINE = r"(?i)ANSWER\s*:\s*([^\n]+)\s*\Z"
