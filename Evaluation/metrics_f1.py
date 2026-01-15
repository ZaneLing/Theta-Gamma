from typing import List

from metrics_gpt35_em import normalize_answer


def answer_f1(pred: str, golds: List[str]) -> float:
    if not pred or not golds:
        return 0.0
    npred = normalize_answer(pred)
    ptoks = set(npred.split())
    for g in golds:
        ng = normalize_answer(g)
        gtoks = set(ng.split())
        if npred == ng:
            return 1.0
        if npred and ng and (npred in ng or ng in npred):
            return 1.0
        if ptoks and gtoks and ptoks & gtoks:
            return 1.0
    return 0.0
