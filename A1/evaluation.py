def compute_micro_f1_score(preds, golds):
    TP, FP, FN = 0, 0, 0

    assert len(preds) == len(golds)

    for pp, gg in zip(preds, golds):
        if gg == 'en':
            if pp != gg:
                FP += 1
                FN += 1
        else:
            if pp != gg:
                FP += 1
                FN += 1
            else:
                TP += 1

    prec = TP / (TP + FP)
    rec =  TP / (TP + FN)
    f1 = (2 * prec * rec) / (prec + rec)

    return f1


def compute_macro_f1_score(preds, golds):
    total = 0
    all_langs = list(set(golds))
    all_langs.remove('en')

    for ln in all_langs:
        TP, FP, FN = 0, 0, 0
        for pp, gg in zip(preds, golds):
            if (gg == ln) and (pp == gg):
                TP += 1
            elif (gg == ln) and (pp != gg):
                FN += 1
            elif (pp == ln) and (pp != gg):
                FP += 1

        prec = TP / (TP + FP)
        rec =  TP / (TP + FN)
        f1 = (2 * prec * rec) / (prec + rec)

        total += f1

    return total / (len(all_langs))