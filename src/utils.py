from collections import defaultdict
import os

class ConfusionMatrix:
    def __init__(self):
        self.seen_class = []
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)

    def add_tp(self, cls):
        if cls not in self.seen_class: self.seen_class.append(cls)
        self.tp[cls] += 1

    def add_fp(self, cls):
        if cls not in self.seen_class: self.seen_class.append(cls)
        self.fp[cls] += 1

    def add_fn(self, cls):
        if cls not in self.seen_class: self.seen_class.append(cls)
        self.fn[cls] += 1

    def precision(self, cls):
        if self.tp[cls] + self.fp[cls] == 0:
            return 0
        return self.tp[cls] / (self.tp[cls] + self.fp[cls])

    def recall(self, cls):
        if self.tp[cls] + self.fn[cls] == 0:
            return 0
        return self.tp[cls] / (self.tp[cls] + self.fn[cls])

    def f_measure(self, cls, beta=1):
        p, r = self.precision(cls), self.recall(cls)
        if p + r == 0:
            return 0
        return (1. + beta*beta) * (r * p) / (beta*beta * p + r)

    def accuracy(self, cls):
        if self.tp[cls] + self.fp[cls] + self.fn[cls] == 0:
            return 0
        return (self.tp[cls]) / (self.tp[cls] + self.fp[cls] + self.fn[cls])

    def macro_precision(self):
        p = 0
        for cls in self.seen_class:
            p += self.precision(cls)
        if len(self.seen_class) == 0:
            return 0
        return p / len(self.seen_class)

    def macro_recall(self):
        r = 0
        for cls in self.seen_class:
            r += self.recall(cls)
        if len(self.seen_class) == 0:
            return 0
        return r / len(self.seen_class)

    def macro_f_measure(self, beta=1):
        f = 0
        for cls in self.seen_class:
            f += self.f_measure(cls, beta)
        if len(self.seen_class) == 0:
            return 0
        return f / len(self.seen_class)

    def macro_accuracy(self):
        acc = 0
        for cls in self.seen_class:
            acc += self.accuracy(cls)
        if len(self.seen_class) == 0:
            return 0
        return acc / len(self.seen_class)

    def micro_precision(self):
        tp, fp = 0, 0
        for cls in self.seen_class:
            tp += self.tp[cls]
            fp += self.fp[cls]
        if tp + fp == 0:
            return 0
        return tp / (tp + fp)

    def micro_recall(self):
        tp, fn = 0, 0
        for cls in self.seen_class:
            tp += self.tp[cls]
            fn += self.fn[cls]
        if tp + fn == 0:
            return 0
        return tp / (tp + fn)

    def micro_f_measure(self, beta=1):
        p, r = self.micro_precision(), self.micro_recall()
        if p + r == 0:
            return 0
        return (1. + beta*beta) * (r * p) / (beta*beta * p + r)

    def micro_accuracy(self):
        t, f = 0, 0
        for cls in self.seen_class:
            t += self.tp[cls]
            f += self.fn[cls] + self.fp[cls]
        if t + f == 0:
            return 0
        return t / (t + f)

    def __str__(self):
        offset = ' ' * 8
        big_line = '\n'
        for cls in self.seen_class:
            tp, fp, fn = self.tp[cls], self.fp[cls], self.fn[cls]
            big_line += offset + f'{cls+":":5} TP: {tp:5}, FP: {fp:5}, FN: {fn:5}\n'
        big_line += '\n'
        for cls in self.seen_class:
            a, p, r, f = self.accuracy(cls), self.precision(cls), self.recall(cls), self.f_measure(cls)
            big_line += offset + f'{cls+":":10} accuracy: {a:.4f}, precision: {p:.4f}, recall: {r:.4f}, f1: {f:.4f}\n'
        big_line += '\n'
        big_line += offset + f'{"MACRO-AVG:":10} accuracy: {self.macro_accuracy():.4f}, precision: {self.macro_precision():.4f}, recall: {self.macro_recall():.4f}, f1: {self.macro_f_measure():.4f}\n'
        big_line += offset + f'{"MICRO-AVG:":10} accuracy: {self.micro_accuracy():.4f}, precision: {self.micro_precision():.4f}, recall: {self.micro_recall():.4f}, f1: {self.micro_f_measure():.4f}'
        return big_line

class EvaluationResult:
    def __init__(self, main_score):
        self.main_score = main_score
        self.metrics = {}

    def add_metric(self, key, value):
        self.metrics[key] = value

    def __str__(self):
        big_line = ""
        for i, key in enumerate(self.metrics.keys()):
            if i: big_line += '\n'
            big_line += f'{key}: {self.metrics[key]}'
        return big_line
