from transformers import LlamaTokenizer, LlamaModel
from torch import nn
import torch
import string
import random

class LinkLlamaModel(nn.Module):
    def __init__(self, char_size):
        super(LinkLlamaModel, self).__init__()

        # Initialize tokenizer and pretrained LLaMA model
        self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.model = LlamaModel.from_pretrained("decapoda-research/llama-7b-hf")

        # Freeze the LLM (no finetuning)
        for param in self.model.parameters():
            param.requires_grad = False

        # Initialize the head (prediction) layer
        self.final_layer = nn.Sequential(
            nn.Linear(768 + char_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid())

    # Process tokens and find pooling indexes only once per tuple
    def _prepare_sample(self, text, signals_characteristics):
        """Process tokens and find pooling indexes

                    Args:
                        text (string) : the full text of the local tuple
                        signals_characteristics (list) :
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

class LinkLlama(object):
    """The full model, containing both LLaMA and the prediction head."""
    def __init__(self, char_size, buffer_size, update_sample_size):

        self.curr_input_ids = None
        self.curr_signal_char_pool = None

        self.sample_buffer_size = buffer_size
        self.sample_buffer_tokid = []
        self.sample_buffer_char = []
        self.sample_buffer_pool = []
        self.sample_buffer_reinforcement = []

        self.batch_sample_size = update_sample_size

        self.linkyLlama = LinkLlamaModel(char_size).to('cuda')
        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.linkyLlama.parameters(), lr=0.001)

    def _get_signal_characteristic_pairs(self, signals_of_interest, sender, tupID):
        signal_list = [splitSignal for signal in signals_of_interest for splitSignal in signal.splitByAttribute()]
        return (signal_list, [sender.featurize_term(signal, tupID) for signal in signal_list])

    # Returns a score
    def predict_raw_terms(self, sender, tupID, data):
        """Uses the full model for inference.

               Args:
                    sender (SenderLlama) : a reference to the Sender--a more compact way to pass terms through.
                        Used for determining candidate terms.
                    tupID (string) : the ID of the tuple we want to predict terms for.
                    data (Datastore) : the underlying data. Used to for tokenization and encoding.

                Returns:
                    list of Signal objects along with their scores.
        """
        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        text = ' '.join(data.table[tupID][1:]).translate(translator)

        self.curr_input_ids, self.curr_signal_char_pool = \
            self.linkyLlama._prepare_sample(text,
                                            self._get_signal_characteristic_pairs(sender.signalIndex[tupID], sender, tupID))

        self.linkyLlama.eval()
        with torch.no_grad():
            results = self.linkyLlama(self.curr_input_ids.unsqueeze(0),
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

        self.linkyLlama.train()
        self.optimizer.zero_grad()

        if split_sample:
            samples = [[x] for x in sample_idx]
        else:
            samples = [sample_idx]

        for samp in samples:
            if len(samp) == 0:
                continue
            pred = self.linkyLlama(
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_tokid[idx] for idx in samp], batch_first=True,
                                                padding_value=1),
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_char[idx] for idx in samp], batch_first=True),
                [self.sample_buffer_pool[idx] for idx in samp])

            loss = self.loss_fn(pred, torch.tensor([self.sample_buffer_reinforcement[i][j] for i in samp for j in
                                                    range(len(self.sample_buffer_reinforcement[i]))],
                                                   dtype=torch.float32, device='cuda').unsqueeze(0)) / len(samples)

            loss.backward()  # call backwards on each one in order to acumulate then gradients, then do a step to make the final backprop step.
            # This will have the same affect as if they were all one batch, but without the hassle (may also be slower)

        torch.nn.utils.clip_grad_norm_(self.linkyLlama.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if torch.isnan(loss):
            print('Loss is NaN')

        # Clear buffer
        while self.sample_buffer_size < len(self.sample_buffer_tokid):
            self.sample_buffer_tokid.pop(0)
            self.sample_buffer_char.pop(0)
            self.sample_buffer_pool.pop(0)
            self.sample_buffer_reinforcement.pop(0)

        return loss.item()
