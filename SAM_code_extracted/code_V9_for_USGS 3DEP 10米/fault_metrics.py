"""Metrics against available labels; they do not establish geological completeness."""
import torch


class FaultMetrics:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0
        self.positive_sum = self.negative_sum = 0.
        self.empty_area = []
        self.samples = 0

    @torch.no_grad()
    def observe(self, prediction, target, weights):
        p, y = prediction[:, :1].float(), target[:, :1] > .5
        valid = torch.broadcast_to(weights, target.shape)[:, :1] > 0
        for pi, yi, vi in zip(p, y, valid):
            if not vi.any():
                continue
            pi, yi = pi[vi], yi[vi]
            predicted = pi >= .5
            self.tp += int((predicted & yi).sum())
            self.fp += int((predicted & ~yi).sum())
            self.fn += int((~predicted & yi).sum())
            self.tn += int((~predicted & ~yi).sum())
            self.positive_sum += float(pi[yi].sum())
            self.negative_sum += float(pi[~yi].sum())
            self.samples += 1
            if not yi.any():
                self.empty_area.append(float(predicted.float().mean()))

    def report(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        n, pos, neg = tp+fp+fn+tn, tp+fn, fp+tn
        ratio = lambda a, b: a/b if b else None
        fraction = ratio(tp+fp, n)
        # Exact threshold collapse is a health check, not a claim about accuracy.
        collapse = ('no_supervision' if not n else
                    'all_positive' if neg and tp+fp == n else
                    'all_negative' if pos and tp+fp == 0 else None)
        return dict(samples=self.samples, pixels=n, tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=ratio(tp, tp+fp), recall=ratio(tp, pos),
                    f1=ratio(2*tp, 2*tp+fp+fn), label_positive_fraction=ratio(pos, n),
                    prediction_positive_fraction=fraction,
                    positive_probability_mean=ratio(self.positive_sum, pos),
                    background_probability_mean=ratio(self.negative_sum, neg),
                    empty_crop_positive_fraction=(sum(self.empty_area)/len(self.empty_area)
                                                  if self.empty_area else None),
                    collapse=collapse, eligible=collapse is None,
                    near_collapse=bool(n and pos and neg and (fraction > .99 or fraction < .001)),
                    scope='Available supervised pixels only; unknown pixels excluded')
