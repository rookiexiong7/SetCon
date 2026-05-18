import re


def find_seg_indices(text):
    all_seg_indices = [m.start() for m in re.finditer(r'\[SEG\]\(([^)]+)\)', text)]
    answer_spans = [(m.start(), m.end()) for m in re.finditer(r'<answer>.*?</answer>', text, re.DOTALL)]
    if len(answer_spans) == 0:
        return [], list(range(len(all_seg_indices)))
    if len(answer_spans) > 1:
        print(f"Warning: There should be only one <answer> tag in the text. {text}")
    start, end = answer_spans[0]

    seg_indices_in_reason = []
    seg_indices_in_answer = []
    for idx, seg_ind in enumerate(all_seg_indices):
        if start <= seg_ind < end:
            seg_indices_in_answer.append(idx)
        elif seg_ind < start:
            seg_indices_in_reason.append(idx)
    return seg_indices_in_reason, seg_indices_in_answer
