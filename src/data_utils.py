import torch
import torch.nn as nn
import pickle

class dictionary:
    def __init__(self):
        self.idx2item = {}
        self.item2idx = {}
        self.size = 0
#         self.add_item('<unk>')

    def add_item(self, item):
        item = item.encode("utf-8")
        if item not in self.item2idx:
            self.item2idx[item] = self.size
            self.idx2item[self.size] = item
            self.size += 1

    def get_index(self, item):
        item = item.encode("utf-8")
        if item in self.item2idx:
            return self.item2idx[item]
        return -1

    def get_indexes(self, items):
        indexes = []
        for item in items:
            indexes.append(self.get_index(item))
        return indexes

    def get_item(self, idx):
        return self.idx2item[idx].decode("utf-8")

    def save(self, path):
        with open(path, "wb") as f:
            state = {"idx2item": self.idx2item, "item2idx": self.item2idx, "size": self.size}
            pickle.dump(mappings, f)

    def load(self, path):
        with open(path, "rb") as f:
            state = pickle.load(f)
            self.idx2item = state["idx2item"]
            self.item2idx = state["item2idx"]
            self.size = state["size"]

    def __len__(self):
        return self.size

    def __str__(self):
        some_tags = "\n".join("  " + str(idx) + "\t" + self.get_item(idx) for idx in range(min(10, len(self)))) + ("\n  ..." if len(self) > 10 else "")
        return f'dictionary with {len(self)} items:\n{some_tags}'

    
def split_tag(chunk_tag):
    """
    split chunk tag into IOBES prefix and chunk_type
    e.g. 
    B-PER -> (B, PER)
    O -> (O, None)
    """
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_start(prev_tag, tag):
    """
    check if a new chunk started between the previous and current word
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    if chunk_type1 != chunk_type2:
        return True
    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_end(prev_tag, tag):
    """
    check if the previous chunk ended between the previous and current word
    e.g. 
    (B-PER, I-PER) -> False
    (B-LOC, O)  -> True
    Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
    this is considered as (B-PER, B-LOC)
    """
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    if chunk_type1 != chunk_type2:
        return True

    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']
