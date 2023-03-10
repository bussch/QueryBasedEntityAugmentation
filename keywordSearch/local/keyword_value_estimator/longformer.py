from transformers import LongformerTokenizer, LongformerModel
from torch import nn
import torch
import string
import random
from torch.optim.lr_scheduler import LambdaLR

class LinkBertModel(nn.Module):
    def __init__(self, useMLP, char_size, finetune):
        super(LinkBertModel, self).__init__()

        self.finetune = finetune

        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

        self.tokenizer.add_tokens(['[BOR]'], special_tokens=True)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if not (self.finetune):
            for param in self.model.parameters():
                param.requires_grad = False

        if useMLP:
            self.final_layer = nn.Sequential(
                nn.Linear(768 + char_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid())
        else:
            self.final_layer = nn.Sequential(
                nn.Linear(768 + char_size, 1)
            )

    # Process tokens and find pooling indexes only once per tuple
    def _prepare_sample(self, sender, text, signals_characteristics, tupID):
        """used to update tuple-specific features as new matches are found.

                    Args:
                        local_tuple_id (string) : the ID of the tuple to update tuple-specific features for
                        external_tuple_ids (list) : a list of string IDs for external tuples for which tuple-specific features
                            should be derived. Expected to be the list of external tuple IDs that match with this local tuple
                            specified by local_tuple_id.
                        external_signal_index (dict) : the signalIndex dictionary of the Receiver. Used to access the external
                            signals associated with the tuple IDS in external_tuple_ids.

                    Data Structures Updated:
                        _has_matched : keeps track of <local_tuple_id: <external_tuple_id: True/False>> to prevent redundant work.
                        if 'external_feat_specific' is True:
                            _max_external_tf (dict) : the maximum external term frequency (count of the term that appears most often)
                                for a external tuple <external_tuple_id: max_TF>. Used for normalization.
                            _external_tf (dict) : mapping of <local_tuple_id: <stemmed_term: <external_tuple_id: TF>>>. A single
                                stemmed_term may appear in multiple matches, so we keep track of each in order to average them.
                        if 'supervised_term_borrowing' is True:
                            signalIndex : adds external signals to the signalIndex
        """

        input_ids = self.tokenizer.encode(' ' + text, truncation=True)

        borrowed_not_found = 0

        tokens_not_found = []
        offset = 0
        pooling = []
        for signal in [x for x in signals_characteristics[0]]:

            # Find all keywords in the tuple that could have produced this stemmed signal
            pooling_subwords = []
            originSet = [tup[0] for tup in signal.originList]
            if not (signal.isLocal):
                originSet.append(signal.keyword)
            for original_word in set(originSet):

                if original_word == None:
                    continue

                # find index(s) of keyword
                sub_ids = self.tokenizer.encode(' ' + original_word)[1:-1]
                if len(sub_ids) == 0:
                    continue

                # Find offsets of all tokens that match the first token of tokenized original_keyword
                for idx in range(len(input_ids)):
                    if sub_ids[0] == input_ids[idx]:  # First sub_token matches at this index...

                        # ...check the remaining sub_tokens. Assume match unless we see a difference or move outside of the tuple's length
                        isWord = True
                        for i in range(len(sub_ids)):
                            if len(input_ids) <= idx + i or sub_ids[i] != input_ids[idx + i]:
                                isWord = False
                        if isWord:
                            pooling_subwords = pooling_subwords + [idx + i for i in range(len(sub_ids))]

            if len(pooling_subwords) == 0:

                if not (signal.isLocal):
                    borrowed_not_found += 1

                tokens_not_found.append(signal.keyword)

                del signals_characteristics[0][offset]
                del signals_characteristics[1][offset]
                continue
            else:
                pooling.append(pooling_subwords)
                offset += 1

        return torch.tensor(input_ids).to('cuda'), (
        signals_characteristics[0], torch.tensor(signals_characteristics[1]).to('cuda'), pooling)

    def forward(self, input_ids, BASE_CHARACTERISTICS, POOLING_INDEXES):

        outputs = self.model(input_ids=input_ids)

        term_score = self.final_layer(
            torch.cat(
                (torch.stack([torch.mean(outputs.last_hidden_state[tup_i][POOLING_INDEXES[tup_i][sig_i]], 0)
                              for tup_i in range(len(outputs.last_hidden_state))
                              for sig_i in range(len(POOLING_INDEXES[tup_i]))]),
                 torch.stack([BASE_CHARACTERISTICS[tup_i][sig_i]
                              for tup_i in range(len(outputs.last_hidden_state))
                              for sig_i in range(len(POOLING_INDEXES[tup_i]))])), 1)).T

        return term_score

class LinkBert(object):
    def __init__(self, char_size, buffer_size, update_sample_size, finetune):

        self.curr_input_ids = None
        self.curr_signal_char_pool = None

        self.useMLP = True  # if False, use linear model

        self.sample_buffer_size = buffer_size
        self.sample_buffer_tokid = []
        self.sample_buffer_char = []
        self.sample_buffer_pool = []
        self.sample_buffer_reinforcement = []

        self.batch_sample_size = update_sample_size

        self.linkyBert = LinkBertModel(self.useMLP, char_size, finetune).to('cuda')
        self.loss_fn = nn.MSELoss()

        if self.useMLP:
            self.optimizer = torch.optim.Adam(self.linkyBert.parameters(), lr=0.001)

            if self.linkyBert.finetune:
                def lr_lambda(current_step: int):
                    num_warmup_steps = 500
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    return 1.0

                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda, last_epoch=-1)

        else:
            self.optimizer = torch.optim.SGD(self.linkyBert.parameters(), lr=0.001, weight_decay=0.0001)

    def is_fitted(self):
        return True

    def get_signal_characteristic_pairs(self, signals_of_interest, sender, tupID):
        signal_list = [splitSignal for signal in signals_of_interest for splitSignal in signal.splitByAttribute()]
        return (signal_list, [sender.featurize_term(signal, tupID, external_feat_specific=sender.config['external_feat_specific']) for signal in signal_list])

    # Returns a score
    def predict_raw_terms(self, sender, tupID, data):

        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        text = ' '.join(data.table[tupID][1:]).translate(translator)

        self.curr_input_ids, self.curr_signal_char_pool = \
            self.linkyBert._prepare_sample(sender, text,
                                           self.get_signal_characteristic_pairs(sender.signalIndex[tupID], sender, tupID),
                                           tupID)

        self.linkyBert.eval()
        with torch.no_grad():
            results = self.linkyBert(self.curr_input_ids.unsqueeze(0),
                                     self.curr_signal_char_pool[1].unsqueeze(0),
                                     [self.curr_signal_char_pool[2]])

        return [(self.curr_signal_char_pool[0][i], results[0][i].item()) for i in range(len(results[0]))]

    def add_sample_to_buffer(self, sample_x_terms, sample_y):
        term_indexes = [self.curr_signal_char_pool[0].index(sample_x_terms[idx]) for idx in range(len(sample_x_terms))]
        self.sample_buffer_tokid.append(self.curr_input_ids)
        self.sample_buffer_char.append(self.curr_signal_char_pool[1][term_indexes])
        self.sample_buffer_pool.append([self.curr_signal_char_pool[2][idx] for idx in term_indexes])
        self.sample_buffer_reinforcement.append(sample_y)

    def partial_fit(self, split_sample):
        sample_idx = random.sample(range(0, len(self.sample_buffer_tokid)),
                                   (len(self.sample_buffer_tokid)
                                    if len(self.sample_buffer_tokid) < self.batch_sample_size
                                    else self.batch_sample_size)
                                   )

        self.linkyBert.train()
        self.optimizer.zero_grad()

        if split_sample:
            samples = [[x] for x in sample_idx]
        else:
            samples = [sample_idx]

        for samp in samples:
            if len(samp) == 0:
                continue
            pred = self.linkyBert(
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_tokid[idx] for idx in samp], batch_first=True,
                                                padding_value=1),
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_char[idx] for idx in samp], batch_first=True),
                [self.sample_buffer_pool[idx] for idx in samp])

            loss = self.loss_fn(pred, torch.tensor([self.sample_buffer_reinforcement[i][j] for i in samp for j in
                                                    range(len(self.sample_buffer_reinforcement[i]))],
                                                   dtype=torch.float32, device='cuda').unsqueeze(0)) / len(samples)

            loss.backward()  # call backwards on each one in order to acumulate then gradients, then do a step to make the final backprop step.
            # This will have the same affect as if they were all one batch, but without the hassle (may also be slower)

        torch.nn.utils.clip_grad_norm_(self.linkyBert.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.linkyBert.finetune:
            self.lr_scheduler.step()

        if torch.isnan(loss):
            print('Loss is NaN')

        # Clear buffer
        while self.sample_buffer_size < len(self.sample_buffer_tokid):
            self.sample_buffer_tokid.pop(0)
            self.sample_buffer_char.pop(0)
            self.sample_buffer_pool.pop(0)
            self.sample_buffer_reinforcement.pop(0)

        return loss.item()
