import torch
import torch.nn as nn
import torch.nn.functional as F
import src.training_utils as training_utils
import numpy as np
from flair.training_utils import store_embeddings
from .data import TagSequence
import src.utils as utils

START_TAG = '<START>'
STOP_TAG = '<STOP>'

class BaseModel(nn.Module):
    def __init__(self, embeddings, hidden_size, tag_dictionary, args):
        super(BaseModel, self).__init__()
        self.word_dropout_rate = args.get('word_dropout', 0.05)
        self.locked_dropout_rate = args.get('locked_dropout', 0.5)
        self.relearn_embeddings = args.get('relearn_embeddings', True)
        self.device = args.get('device', torch.device('cuda') if torch.cuda.is_available() else 'cpu')
        self.use_crf = args.get('use_crf', True)
        self.cell_type = args.get('cell_type', 'LSTM') # in (RNN, LSTM, GRU)
        self.tag_type_mode = args.get('tag_name', 'ner')
        self.bidirectional = args.get('bidirectional', True)

        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.tag_dictionary = tag_dictionary
        self.tagset_size = len(tag_dictionary)

        self.word_dropout = WordDropout(self.word_dropout_rate)
        self.locked_dropout = LockedDropout(self.locked_dropout_rate)

        if self.relearn_embeddings:
            self.embedding2nn = torch.nn.Linear(embeddings.embedding_length, embeddings.embedding_length)

        self.rnn = getattr(torch.nn, self.cell_type)(embeddings.embedding_length, hidden_size, 
                                          batch_first=True, bidirectional=self.bidirectional)
        self.linear = nn.Linear((2 if self.bidirectional else 1) * self.hidden_size, self.tagset_size)
        if self.use_crf:
            self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
            self.transitions.detach()[self.tag_dictionary.get_index(START_TAG), :] = -10000
            self.transitions.detach()[:, self.tag_dictionary.get_index(STOP_TAG)] = -10000

        self.to(self.device)

    def _forward(self, sentences):
        self.embeddings.embed(sentences)
        lens = [len(sentence.tokens) for sentence in sentences]
        embeddings_list = []
        for sentence in sentences:
            embeddings_list.append(torch.cat([token.get_embedding().unsqueeze(0).to(self.device) for token in sentence.tokens], 0))
        x = training_utils.pad_sequence(embeddings_list, batch_first=True, padding_value=0, padding='post')
        x = self.word_dropout(x)
        x = self.locked_dropout(x)
        if self.relearn_embeddings:
            x = self.embedding2nn(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False, batch_first=True)
        rnn_output, hidden = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        x = self.locked_dropout(x)
        x = self.linear(x)
        return x

    def _scores(self, x, tag_sequences, reduce_to_batch=False):
        tags_list = []
        for tag_sequence in tag_sequences:
            tags_list.append(torch.tensor([self.tag_dictionary.get_index(tag) for tag in tag_sequence], dtype=torch.long, device=self.device))
        lens = [len(tag_sequence) for tag_sequence in tag_sequences]
        if self.use_crf:
            y = training_utils.pad_sequence(tags_list, batch_first=True, padding_value=0, padding='post')
            forward_score = self._forward_alg(x, lens)
            gold_score = self._score_sentence(x, y, lens)
            return gold_score, forward_score # (batch_size,), (batch_size,)
        else:
            score = torch.zeros(x.shape[0], x.shape[1]).to(self.device)
            for i, feats, tags, length in zip(range(x.shape[0]), x, tags_list, lens):
                feats = feats[:length]
                for j in range(feats.shape[0]):
                    score[i][j] = torch.nn.functional.cross_entropy(feats[j:j+1], tags[j:j+1])
            if reduce_to_batch:
                reduced_score = torch.zeros(score.shape[0]).to(self.device)
                for i in range(reduced_score.shape[0]):
                    reduced_score[i] = score[i, :lens[i]].sum()
                score = reduced_score
            return score # (batch_size, max_time) or (batch_size)

    def _loss(self, x, tag_sequences):
        if self.use_crf:
            gold_score, forward_score = self._scores(x, tag_sequences)
            score = forward_score - gold_score
            return score.mean()
        else:
            entropy_score = self._scores(x, tag_sequences, reduce_to_batch=True)
            return entropy_score.mean()

    def evaluate(self, data_loader, embedding_storage_mode):
        with torch.no_grad():
            eval_loss = 0
            if self.use_crf:
                transitions = self.transitions.detach().cpu().numpy()
            else:
                transitions = None
            cm = utils.ConfusionMatrix()
            for batch in data_loader:
                sentences, tag_sequences = batch
                x = self._forward(sentences)
                loss = self._loss(x, tag_sequences)
                predicted_tag_sequences, confs = self._obtain_labels(x, sentences, transitions, self.tag_type_mode)
                for i in range(len(sentences)):
                    gold = tag_sequences[i].get_span()
                    pred = predicted_tag_sequences[i].get_span()
                    for pred_span in pred:
                        if pred_span in gold:
                            cm.add_tp(pred_span[0])
                        else:
                            cm.add_fp(pred_span[0])
                    for gold_span in gold:
                        if gold_span not in pred:
                            cm.add_fn(gold_span[0])
                eval_loss += loss.item()
                store_embeddings(sentences, embedding_storage_mode)

            eval_loss /= len(data_loader)
            if self.tag_type_mode == 'ner':
                res = utils.EvaluationResult(cm.micro_f_measure())
            else:
                res = utils.EvaluationResult(cm.micro_accuracy)
            res.add_metric('Confusion Matrix', cm)
            return eval_loss, res

    def _forward_alg(self, feats, lens_):
        init_alphas = torch.FloatTensor(self.tagset_size).fill_(-10000.0)
        init_alphas[self.tag_dictionary.get_index(START_TAG)] = 0.
        forward_var = torch.zeros(
            feats.shape[0],
            feats.shape[1] + 1,
            feats.shape[2],
            dtype=torch.float,
            device=self.device,
        )
        forward_var[:, 0, :] = init_alphas[None, :].repeat(feats.shape[0], 1)
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)
        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[:, i, :][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )
            max_tag_var, _ = torch.max(tag_var, dim=2)
            tag_var = tag_var - max_tag_var[:, :, None].repeat(
                1, 1, transitions.shape[2]
            )
            agg_ = torch.log(torch.sum(torch.exp(tag_var), dim=2))
            cloned = forward_var.clone()
            cloned[:, i + 1, :] = max_tag_var + agg_
            forward_var = cloned
        forward_var = forward_var[range(forward_var.shape[0]), lens_, :]
        terminal_var = forward_var + self.transitions[
            self.tag_dictionary.get_index(STOP_TAG)
        ][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags_idx, lens_):
        start = torch.tensor(
            [self.tag_dictionary.get_index(START_TAG)], device=self.device
        )
        start = start[None, :].repeat(tags_idx.shape[0], 1)
        stop = torch.tensor(
            [self.tag_dictionary.get_index(STOP_TAG)], device=self.device
        )
        stop = stop[None, :].repeat(tags_idx.shape[0], 1)
        pad_start_tags = torch.cat([start, tags_idx], 1)
        pad_stop_tags = torch.cat([tags_idx, stop], 1)
        for i in range(len(lens_)):
            pad_stop_tags[i, lens_[i] :] = self.tag_dictionary.get_index(
                STOP_TAG
            )
        score = torch.FloatTensor(feats.shape[0]).to(self.device)
        for i in range(feats.shape[0]):
            r = torch.LongTensor(range(lens_[i])).to(self.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lens_[i] + 1], pad_start_tags[i, : lens_[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags_idx[i, : lens_[i]]]) 
            # for mixup ->  emission scores   --> sum over j <= lens_[i] ( lambda[j] * f[i, j, tags_idx[i, j]] ) 
            #               transition scores -->  t[tags_idx[i, j], tags_idx[i, j+1]] * (lambda[j] + lambda[j+1]) / 2.0
        return score

    def _viterbi_decode(self, feats, transitions):
        id_start = self.tag_dictionary.get_index(START_TAG)
        id_stop = self.tag_dictionary.get_index(STOP_TAG)
        backpointers = np.empty(shape=(feats.shape[0], self.tagset_size), dtype=np.int_)
        backscores = np.empty(shape=(feats.shape[0], self.tagset_size), dtype=np.float32)
        init_vvars = np.expand_dims(np.repeat(-10000.0, self.tagset_size), axis=0).astype(np.float32)
        init_vvars[0][id_start] = 0
        forward_var = init_vvars
        for index, feat in enumerate(feats):
            next_tag_var = forward_var + transitions
            bptrs_t = next_tag_var.argmax(axis=1)
            viterbivars_t = next_tag_var[np.arange(bptrs_t.shape[0]), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores[index] = forward_var
            forward_var = forward_var[np.newaxis, :]
            backpointers[index] = bptrs_t
        terminal_var = forward_var.squeeze() + transitions[id_stop]
        terminal_var[id_stop] = -10000.0
        terminal_var[id_start] = -10000.0
        best_tag_id = terminal_var.argmax()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == id_start
        best_path.reverse()
        def _softmax(x, axis):
            x_norm = x - x.max(axis=axis, keepdims=True)
            y = np.exp(x_norm)
            return y / y.sum(axis=axis, keepdims=True)
        best_scores_softmax = _softmax(backscores, axis=1)
        best_scores_np = np.max(best_scores_softmax, axis=1)
        return best_scores_np.tolist(), best_path

    def _obtain_labels(self, feats, sentences, transitions, tag_type_mode):
        lengths = [len(sentence.tokens) for sentence in sentences]
        tag_sequences = []
        confs = []
        feats = feats.cpu()
        if self.use_crf:
            feats = feats.numpy()
        else:
            for index, length in enumerate(lengths):
                feats[index, length:] = 0
            softmax_batch = F.softmax(feats, dim=2).cpu()
            scores_batch, prediction_batch = torch.max(softmax_batch, dim=2)
            feats = zip(scores_batch, prediction_batch)
        for feats, length in zip(feats, lengths):
            if self.use_crf:
                confidences, tag_seq = self._viterbi_decode(feats=feats[:length], transitions=transitions)
            else:
                score, prediction = feats
                confidences = score[:length].tolist()
                tag_seq = prediction[:length].tolist()
            tag_sequences.append(TagSequence([self.tag_dictionary.get_item(tag_idx) for tag_idx in tag_seq], mode=tag_type_mode))
            confs.append([conf for conf in confidences])
        return tag_sequences, confs


class Normal(BaseModel):
    def __init__(self, embeddings, hidden_size, tag_dictionary, **args):
        super(Normal, self).__init__(embeddings, hidden_size, tag_dictionary, args)

    def forward_loss(self, batch):
        sentences, tags = batch
        x = self._forward(sentences)
        return self._loss(x, tags)

class PreOutputMixup(BaseModel):
    def __init__(self, embeddings, hidden_size, tag_dictionary, **args):
        super(PreOutputMixup, self).__init__(embeddings, hidden_size, tag_dictionary, args)
        self.sort = args.get('sort', True)
        self.lambdas_generator = args.get('lambdas_generator', "beta") + "_lambdas_generator"
        self.lambdas_generator_params = args.get('lambdas_generator_params')

    def forward(self, sentences):
        self.embeddings.embed(sentences)
        lens = [len(sentence.tokens) for sentence in sentences]
        embeddings_list = []
        for sentence in sentences:
            embeddings_list.append(torch.cat([token.get_embedding().unsqueeze(0).to(self.device) for token in sentence.tokens], 0))
        x = training_utils.pad_sequence(embeddings_list, batch_first=True, padding_value=0, padding='post')
        x = self.word_dropout(x)
        x = self.locked_dropout(x)
        if self.relearn_embeddings:
            x = self.embedding2nn(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False, batch_first=True)
        rnn_output, hidden = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        x = self.locked_dropout(x)
        return x

    def forward_loss(self, batch1, batch2):
        if self.sort: # sort batches by their length
            batch1 = training_utils.sort(batch1)
            batch2 = training_utils.sort(batch2)
        h1 = self.forward(batch1[0]) # (batch, length1, hidden_size)
        h2 = self.forward(batch2[0]) # (batch, length2, hidden_size)
        mx_len =  max(h1.shape[1], h2.shape[1])
        lens1 = [len(sentence.tokens) for sentence in batch1[0]]
        lens2 = [len(sentence.tokens) for sentence in batch2[0]]
        lens = [max(l1, l2) for l1, l2 in zip(lens1, lens2)]
        z = torch.zeros(h1.shape[0], mx_len-min(h1.shape[1], h2.shape[1]), h1.shape[2]).to(self.device)
        if h1.shape[1] < mx_len:
            h1 = torch.cat((h1, z), dim=1)
        else:
            h2 = torch.cat((h2, z), dim=1)
        if self.lambdas_generator == "beta_lambdas_generator":    
            lambdas = getattr(training_utils, self.lambdas_generator)(self.lambdas_generator_params['alpha'],
                                                                      self.lambdas_generator_params['beta'],
                                                                      h1.shape[0],
                                                                      h1.shape[1],
                                                                      (h1.shape[2],),
                                                                      self.lambdas_generator_params['rho']
                                                                      )

        lambdas = torch.from_numpy(lambdas).type(torch.float32).to(self.device) #(batch_size, mx_len, hidden_size)
        
        h = lambdas * h1 + (1 - lambdas) * h2
        x = self.linear(h)
        if self.use_crf:
            lambdas = torch.FloatTensor([lambdas[i, :l, 0].mean() for i, l in enumerate(lens)]).to(self.device)
            gold_1, forward_1 = self._scores(x, batch1[1])
            gold_2, forward_2 = self._scores(x, batch2[1])
            score = lambdas * (forward_1 - gold_1) + (1 - lambdas) * (forward_2 - gold_2)
        else:
            score_1 = self._scores(x, batch1[1], reduce_to_batch=False)            
            score_2 = self._scores(x, batch2[1], reduce_to_batch=False)
            score = torch.zeros(score_1.shape[0]).to(self.device)
            for i, l in enumerate(lens):
                score[i] = (lambdas[i, :l, 0] * score_1[i, :l] + (1 - lambdas[i, :l, 0]) * score_2[i, :l]).sum()
        return score.mean()

class ThroughTimeMixup(BaseModel):
    def __init__(self, embeddings, hidden_size, tag_dictionary, **args):
        super(ThroughTimeMixup, self).__init__(embeddings, hidden_size, tag_dictionary, args)
        self.sort = args.get('sort', True)
        self.lambdas_generator = args.get('lambdas_generator', "beta") + "_lambdas_generator"
        self.lambdas_generator_params = args.get('lambdas_generator_params')
        del(self.rnn)
        self.forward_cell = getattr(torch.nn, self.cell_type + "Cell")(embeddings.embedding_length, self.hidden_size).to(self.device)
        if self.bidirectional:
            self.backward_cell = getattr(torch.nn, self.cell_type + "Cell")(embeddings.embedding_length, self.hidden_size).to(self.device)

    def forward(self, sentences1, sentences2, lambdas):
        sentences = sentences1 + sentences2
        self.embeddings.embed(sentences)
        embeddings_list = []
        for sentence in sentences:
            embeddings_list.append(torch.cat([token.get_embedding().unsqueeze(0).to(self.device) for token in sentence.tokens], 0))            
        x = training_utils.pad_sequence(embeddings_list, batch_first=True, padding_value=0, padding='post')
        x = self.word_dropout(x)
        x = self.locked_dropout(x)
        if self.relearn_embeddings:
            x = self.embedding2nn(x)
        x1 = x[:len(sentences1)]
        x2 = x[len(sentences1):]
        h_forward = self._forward_cell(x1, x2, lambdas)
        if self.bidirectional:
            h_backward = self._forward_cell(x1, x2, lambdas, backward=True)
            h = torch.cat([h_forward, h_backward], 2).to(self.device)
        else:
            h = h_forward
        x = self.locked_dropout(h)
        x = self.linear(x)
        return x

    def _forward_cell(self, x1, x2, lambdas, backward=False):
        h = torch.zeros(x1.shape[0], self.hidden_size).to(self.device)
        if self.cell_type == 'LSTM':
            c = torch.zeros(x1.shape[0], self.hidden_size).to(self.device)
        _h = torch.zeros(x1.shape[0], x1.shape[1], self.hidden_size).to(self.device)
        assert(x1.shape[1] == x2.shape[1])
        it = range(x1.shape[1]-1, -1, -1) if backward else range(x1.shape[1])
        cell = self.backward_cell if backward else self.forward_cell
        for t in it:                
            if self.cell_type == 'LSTM':
                h_1, c_1 = cell(x1[:, t], (h, c))
                h_2, c_2 = cell(x2[:, t], (h, c))
                h = lambdas[:, t] * h_1 + (1 - lambdas[:, t]) * h_2
                c = lambdas[:, t] * c_1 + (1 - lambdas[:, t]) * c_2
            else:
                h_1 = cell(x1[:, t], h)
                h_2 = cell(x2[:, t], h)
                h = lambdas[:, t] * h_1 + (1 - lambdas[:, t]) * h_2
            _h[:, t] = h
        return _h

    def forward_loss(self, batch1, batch2):
        if self.sort:
            batch1 = training_utils.sort(batch1)
            batch2 = training_utils.sort(batch2)
        mx_len = max([len(a) for a in batch1[0] + batch2[0]])
        lens1 = [len(sentence.tokens) for sentence in batch1[0]]
        lens2 = [len(sentence.tokens) for sentence in batch2[0]]
        lens = [max(l1, l2) for l1, l2 in zip(lens1, lens2)]
        if self.lambdas_generator == "beta_lambdas_generator":
            lambdas = getattr(training_utils, self.lambdas_generator)(self.lambdas_generator_params['alpha'],
                                                                      self.lambdas_generator_params['beta'],
                                                                      len(batch1[0]), 
                                                                      mx_len,
                                                                      (self.hidden_size,),
                                                                      self.lambdas_generator_params['rho'],
                                                                      )
            
        lambdas = torch.from_numpy(lambdas).type(torch.float32).to(self.device) # (batch_size, mx_len, hidden_size)
        x = self.forward(batch1[0], batch2[0], lambdas) # (bs, time, tag_size)
        if self.use_crf:
            lambdas = torch.FloatTensor([lambdas[i, :l, 0].mean() for i, l in enumerate(lens)]).to(self.device)
            gold_1, forward_1 = self._scores(x, batch1[1])
            gold_2, forward_2 = self._scores(x, batch2[1])
            score = lambdas * (forward_1 - gold_1) + (1 - lambdas) * (forward_2 - gold_2)
        else:
            score_1 = self._scores(x, batch1[1], reduce_to_batch=False)
            score_2 = self._scores(x, batch2[1], reduce_to_batch=False)
            score = torch.zeros(score_1.shape[0]).to(self.device)
            for i, l in enumerate(lens):
                score[i] = (lambdas[i, :l, 0] * score_1[i, :l] + (1 - lambdas[i, :l, 0]) * score_2[i, :l]).sum()
        return score.mean()

    def _forward(self, sentences):
        self.embeddings.embed(sentences)
        embeddings_list = []
        for sentence in sentences:
            embeddings_list.append(torch.cat([token.get_embedding().unsqueeze(0).to(self.device) for token in sentence.tokens], 0))
        x = training_utils.pad_sequence(embeddings_list, batch_first=True, padding_value=0, padding='post')
        x = self.word_dropout(x)
        x = self.locked_dropout(x)
        if self.relearn_embeddings:
            x = self.embedding2nn(x)
        h_forward = torch.zeros(x.shape[0], x.shape[1], self.hidden_size).to(self.device)
        h_f = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
        if self.cell_type == 'LSTM':
            c_f = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
        for t in range(x.shape[1]):
            if self.cell_type == 'LSTM':
                h_f, c_f = self.forward_cell(x[:, t], (h_f, c_f))
            else:
                h_f = self.forward_cell(x[:, t], h_f)
            h_forward[:, t] = h_f
        if self.bidirectional:
            h_backward = torch.zeros(x.shape[0], x.shape[1], self.hidden_size).to(self.device)
            h_b = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
            c_b = torch.zeros(x.shape[0], self.hidden_size).to(self.device)
            for t in range(x.shape[1]-1, -1, -1):
                if self.cell_type == 'LSTM':
                    h_b, c_b = self.backward_cell(x[:, t], (h_b, c_b))
                else:
                    h_b = self.backward_cell(x[:, t], h_b)
                h_backward[:, t] = h_b
            h = torch.cat([h_forward, h_backward], 2).to(self.device)
        else:
            h = h_forward
        x = self.locked_dropout(h)
        x = self.linear(x)
        return x

class InputMixup(BaseModel):
    def __init__(self, embeddings, hidden_size, tag_dictionary, **args):
        super(InputMixup, self).__init__(embeddings, hidden_size, tag_dictionary, args)
        self.sort = args.get('sort', True)
        self.lambdas_generator = args.get('lambdas_generator', "beta") + "_lambdas_generator"
        self.lambdas_generator_params = args.get('lambdas_generator_params')

    def forward_loss(self, batch1, batch2):
        if self.sort:
            batch1 = training_utils.sort(batch1)
            batch2 = training_utils.sort(batch2)
        sentences = batch1[0] + batch2[0]
        self.embeddings.embed(sentences)
        lens = [max(len(batch1[0][i].tokens), len(batch2[0][i].tokens)) for i in range(len(batch1[0]))]
        embeddings_list = []
        for sentence in sentences:
            embeddings_list.append(torch.cat([token.get_embedding().unsqueeze(0).to(self.device) for token in sentence.tokens], 0))
        x = training_utils.pad_sequence(embeddings_list, batch_first=True, padding_value=0, padding='post')
#         (2*batch, time, embedding_size)
        if self.lambdas_generator == "beta_lambdas_generator":
            lambdas = getattr(training_utils, self.lambdas_generator)(self.lambdas_generator_params['alpha'],
                                                                      self.lambdas_generator_params['beta'],
                                                                      int(x.shape[0]//2),
                                                                      x.shape[1], 
                                                                      (x.shape[2],),
                                                                      self.lambdas_generator_params['rho'],
                                                                      )
        lambdas = torch.from_numpy(lambdas).type(torch.float32).to(self.device) #(batch, time, embedding_size)

        x = lambdas * x[:len(batch1[0])] + (1 - lambdas) * x[len(batch1[0]):]
        x = self.word_dropout(x)
        x = self.locked_dropout(x)
        if self.relearn_embeddings:
            x = self.embedding2nn(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(x, lens, enforce_sorted=False, batch_first=True)
        rnn_output, hidden = self.rnn(packed)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        x = self.locked_dropout(x)
        x = self.linear(x)
        if self.use_crf:
            lambdas = torch.FloatTensor([lambdas[i, :l, 0].mean() for i, l in enumerate(lens)]).to(self.device)
            gold_1, forward_1 = self._scores(x, batch1[1])
            gold_2, forward_2 = self._scores(x, batch2[1])
            score = lambdas * (forward_1 - gold_1) + (1 - lambdas) * (forward_2 - gold_2)
        else:
            score_1 = self._scores(x, batch1[1], reduce_to_batch=False)
            score_2 = self._scores(x, batch2[1], reduce_to_batch=False)
            score = torch.zeros(score_1.shape[0]).to(self.device)
            for i, l in enumerate(lens):
                score[i] = (lambdas[i, :l, 0] * score_1[i, :l] + (1 - lambdas[i, :l, 0]) * score_2[i, :l]).sum()
        return score.mean()

def log_sum_exp_batch(vecs):
    maxi = torch.max(vecs, 1)[0]
    maxi_bc = maxi[:, None].repeat(1, vecs.shape[1])
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), 1))
    return maxi + recti_


class LockedDropout(torch.nn.Module):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, dropout_rate=0.5, batch_first=True, inplace=False):
        super(LockedDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.batch_first = batch_first
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout_rate)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout_rate)
        mask = mask.expand_as(x)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)


class WordDropout(torch.nn.Module):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def __init__(self, dropout_rate=0.05, inplace=False):
        super(WordDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.inplace = inplace

    def forward(self, x):
        if not self.training or not self.dropout_rate:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.dropout_rate)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x

    def extra_repr(self):
        inplace_str = ", inplace" if self.inplace else ""
        return "p={}{}".format(self.dropout_rate, inplace_str)



