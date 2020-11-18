import numpy as np
import os
import src.data_utils as data_utils
import src.logger as logger
from torch.utils import data
import copy
from flair.data import Sentence
from collections import defaultdict

class TagSequence:
    def __init__(self, tags, mode='none'):
        self.tags = tags
        if mode == 'ner':
            self.span = []
            prev = 'O'
            flag = False
            tmp = ['Type', 'start', 'end']
            for i, tag in enumerate(tags):
                _, _type = data_utils.split_tag(tag)
                if flag:
                    if data_utils.is_chunk_end(prev, tag):
                        flag = False
                        tmp[2] = i-1
                        self.span.append(copy.copy(tmp))
                if data_utils.is_chunk_start(prev, tag):
                    flag = True
                    tmp[0] = _type
                    tmp[1] = i
                prev = tag
            if flag:
                tmp[2] = len(tags)-1
                self.span.append(copy.copy(tmp))

    def get_span(self):
        return self.span

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, i):
        return self.tags[i]

    def __str__(self):
        return "Tags: \"" + " ".join([tag for tag in self.tags]) + f"\" - {len(self.tags)} Tags"

    def __repr__(self):
        return "Tags: \"" + " ".join([tag for tag in self.tags]) + f"\" - {len(self.tags)} Tags"

class DataLoader(data.DataLoader):
    def __init__(self, dataset, **args):
        def collate_fn(batch):
            text = [entry[0] for entry in batch]
            label =[entry[1] for entry in batch]
            return text, label
        super(DataLoader, self).__init__(dataset, collate_fn=collate_fn, **args)

class Dataset(data.Dataset):
    def __init__(self, has_tags):
        self._has_tags = has_tags
        self.sentences = []
        self.add_sentence = lambda sentence: self.sentences.append(sentence)
        if has_tags:
            self.tag_sequences = []
            self.add_tag_sequence = lambda tag_seq: self.tag_sequences.append(tag_seq)
        else:
            self.labels = []
            self.add_label = lambda label: self.labels.append(label)

    def __add__(self, dataset):
        if dataset._has_tags != self._has_tags:
            raise TypeError('Datasets cannot be added!')
        new_dataset = Dataset(self._has_tags)
        new_dataset.sentences = copy.copy(self.sentences)
        new_dataset.sentences.extend(dataset.sentences)
        if self._has_tags:
            new_dataset.tag_sequences = copy.copy(self.tag_sequences)
            new_dataset.tag_sequences.extend(dataset.tag_sequences)
        else:
            new_dataset.labels = copy.copy(self.labels)
            new_dataset.labels.extend(dataset.labels)
        return new_dataset

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return [self.sentences[i], self.tag_sequences[i] if self._has_tags else self.labels[i]]

class Corpus:
    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None

    def create_tags_dictionary(self):
        dic = data_utils.dictionary()
        for i in range(len(self.train)):
            for j in range(len(self.train.tag_sequences[i])):
                dic.add_item(self.train.tag_sequences[i][j])
        dic.add_item('<START>')
        dic.add_item('<STOP>')
        return dic

    def create_label_dictionary(self):
        pass

class Conll_Corpus(Corpus):
    def __init__(self, path, tag_name, tag_scheme=None):
        super(Conll_Corpus, self).__init__()
        self.path = path
        self.tag_name = tag_name
        self.tag_scheme = tag_scheme
        self.tag2column = {'pos': 1, 'chunk': 2, 'ner': 3}

    def build(self, train='eng.train', dev='eng.testa', test='eng.testb'):
        logger.log(f'Reading data from {self.path}')
        self.train = self.get(train)
        logger.log(f'Train {self.path+"/"+train}')
        self.dev = self.get(dev)
        logger.log(f'Dev {self.path+"/"+dev}')
        self.test  = self.get(test)
        logger.log(f'Test {self.path+"/"+test}')

    def get(self, fname):
        with open(os.path.join(self.path, fname), 'r') as f:
            lines = f.readlines()
            dataset = Dataset(has_tags=True)
            sentence, tags = [], []
            tag_column = self.tag2column[self.tag_name]
            for i, line in enumerate(lines):
                line = line.split()
                if not line:
                    if sentence[0] == '-DOCSTART':
                        continue
                    if self.tag_scheme == 'iobes':
                        iob2(tags)
                        tags = iob_iobes(tags)
                    dataset.add_sentence(Sentence(' '.join(token for token in sentence)))
                    dataset.add_tag_sequence(TagSequence(tags, mode='ner'))
                    sentence = []
                    tags = []
                else:
                    word = line[0]
                    tag = line[tag_column]
                    sentence.append(word)
                    tags.append(tag)
        return dataset

class imdb_reader:
    #----------------------------------------------------------------------
    # --------------------------- should change ---------------------------
    #----------------------------------------------------------------------
    def __init__(self, path, fname):
        self.path = path
        self.fname = fname
        self.normalized_fn = 'normalized_texts.txt'
        self.label_fn = 'labels.txt'

        if (not os.path.exists(os.path.join(self.path, self.normalized_fn))) or (not os.path.exists(os.path.join(self.path, self.label_fn))):
            data = pd.read_csv(os.path.join(self.path, self.fname))
            texts = data['review'].tolist()
            labels = data['sentiment'].tolist()
            self.preprocess_and_save(texts, labels)

    def get(self, valid_ratio, test_ratio):
        texts_f  = open(os.path.join(self.path, self.normalized_fn), 'r')
        labels_f = open(os.path.join(self.path, self.label_fn), 'r')
        texts = texts_f.readlines()
        labels = labels_f.readlines()
        n = len(texts)
        cut1 = int((1-valid_ratio-test_ratio) * n)
        cut2 = int((1-test_ratio) * n)
        w_train, w_valid, w_test = texts[:cut1], texts[cut1:cut2], texts[cut2:]
        l_train, l_valid, l_test = labels[:cut1], labels[cut1:cut2], labels[cut2:]
        w_train = [_.split() for _ in w_train]
        w_valid = [_.split() for _ in w_valid]
        w_test  = [_.split() for _ in w_test ]
        l_train = [_.rstrip('\n') for _ in l_train]
        l_valid = [_.rstrip('\n') for _ in l_valid]
        l_test = [_.rstrip('\n') for _ in l_test]
        return w_train, w_valid, w_test, l_train, l_valid, l_test

    def preprocess_and_save(self, texts, labels):
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        stopwords = set(nltk.corpus.stopwords.words('english'))
        # remove punctuation, digits and stopwords & to lower
        new_texts = [[w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stopwords] for text in texts]
        # verbs to origin
        new_texts = [[wordnet_lemmatizer.lemmatize(w, 'v') if pos[0] == 'V' else w for w, pos in nltk.pos_tag(text)] for text in new_texts]
        f = open(os.path.join(self.path, self.normalized_fn), 'w')
        g = open(os.path.join(self.path, self.label_fn), 'w')
        f.write(' '.join(new_texts[0]))
        g.write(labels[0])
        for i in range(1, len(new_texts)):
            f.write('\n')
            f.write(' '.join(new_texts[i]))
            g.write('\n')
            g.write(labels[i])
        f.close()
        g.close()

class pbt_reader:
    #----------------------------------------------------------------------
    # --------------------------- should change ---------------------------
    #----------------------------------------------------------------------
    def __init__(self, path):
        self.path = path

    def get(self, fname):
        f = open(os.path.join(self.path, fname), 'r')
        lines = f.readlines()
        sentences = []
        for i, line in enumerate(lines):
            sentences.append(line.split())
        f.close()
        return sentences

def iob2(tags):
    """
    Check that tags have a valid BIO format.
    Tags in BIO1 format are converted to BIO2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True

def iob_iobes(tags):
    """
    the function is used to convert
    BIO -> BIOES tagging
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
