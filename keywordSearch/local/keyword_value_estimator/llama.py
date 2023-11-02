from transformers import LlamaTokenizer, LlamaModel
from torch import nn
import torch
import string
import random
from additionalScripts.output_config import OutputConfig

DEVICE = 'cuda'

class LinkLlamaModel(nn.Module):
    def __init__(self, char_size, config, dataset):
        super(LinkLlamaModel, self).__init__()

        self.config = config

        self.device = DEVICE

        auth_token = OutputConfig(self.config['config_path']).paths['hf_auth_token']

        # Initialize tokenizer and pretrained LLaMA model
        self.tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
        self.language_model = LlamaModel.from_pretrained("decapoda-research/llama-7b-hf")

        token_dim = 4096

        # *** Adjusted from https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py. Note that this is not DEEP
        #   (i.e., prefixes are only learned for the first layer of the transformer)
        self.prefix_seq_len = self.config['lm_prefixes']
        if self.prefix_seq_len > 0:
            self.prefix_tokens = torch.arange(self.prefix_seq_len).long().to(self.device)
            self.prefix_encoder = nn.Embedding(self.prefix_seq_len, token_dim).to(self.device)

        # attribute_encodings
        self.attribute_embeddings = {attr: i+1 for i, attr in enumerate(dataset.header[1:])}
        self.padding_index = 0
        self.attribute_encoder = nn.Embedding(len(self.attribute_embeddings)+1, token_dim, padding_idx=0).to(self.device)

        # Freeze the LLM (no finetuning)
        for param in self.model.parameters():
            param.requires_grad = False

        # For "Effective Entity Augmentation By Querying External Data Sources"
        self.final_layer = nn.Sequential(
            nn.Linear(token_dim + char_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid())

        # # For "Generating Data Augmentation Queries Using Large Language Models"
        # self.final_layer = nn.Sequential(
        #     nn.Linear(token_dim + char_size, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Sigmoid())

    def _prepare_sample(self, data, tupID, signals_characteristics):
        """Process tokens and find pooling indexes

                    Args:
                        text (string) : the full text of the local tuple
                        signals_characteristics (list) :
        """

        input_ids = []
        attribute_ids = []
        translator = str.maketrans('', '', string.punctuation.replace('-', ''))
        self.TOKEN_TEST = []
        for attr, cell in zip(data.header[1:], data.table[tupID][1:]):
            text = cell.translate(translator)

            segment_length = len(input_ids)
            input_ids = input_ids + self.tokenizer.encode(' ' + text, add_special_tokens=False)

            segment_length = len(input_ids) - segment_length
            attribute_ids = attribute_ids + [self.attribute_embeddings[attr]]*segment_length

        input_ids = [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        attribute_ids = [self.padding_index] + attribute_ids + [self.padding_index]

        offset = 0
        pool_indexes = []
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
                sub_ids = self.tokenizer.encode(' ' + original_word, add_special_tokens=False)
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
                            pooling_subwords = pooling_subwords + [idx + i for i in range(len(sub_ids))] # Pool over all subwords

            if len(pooling_subwords) == 0:
                del signals_characteristics[0][offset]
                del signals_characteristics[1][offset]
                continue
            else:
                pool_indexes.append(pooling_subwords)
                offset += 1

        return torch.tensor(input_ids).to(self.device), (
            signals_characteristics[0], torch.tensor(signals_characteristics[1]).to(self.device), pool_indexes), \
               torch.tensor(attribute_ids, dtype=torch.long).to(self.device)

    def _get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        return past_key_values

    def forward(self, input_ids, BASE_CHARACTERISTICS, POOLING_INDEXES, attribute_ids):

        batch_size = input_ids.shape[0]
        inputs_embeds = self.language_model.embeddings.word_embeddings(input_ids)

        if self.prefix_seq_len > 0:
            # Insert prompt after beginning-of-sentence token
            inputs_embeds = torch.cat((inputs_embeds[:, :1], self._get_prompt(batch_size=batch_size), inputs_embeds[:, 1:]), dim=1)
            attribute_ids = torch.cat(
                (torch.full((batch_size, self.prefix_seq_len), self.padding_index, dtype=torch.long).to(self.device), attribute_ids), dim=1)

        if self.config['attribute_encoding']:
            inputs_embeds = inputs_embeds + self.attribute_encoder(attribute_ids)

        # Push through language model
        outputs = self.language_model(inputs_embeds=inputs_embeds)

        # Remove prefixes to ensure proper pooling
        outputs.last_hidden_state = outputs.last_hidden_state[:, self.prefix_seq_len:]
        pooled_hidden_states = torch.stack([torch.mean(outputs.last_hidden_state[tup_i][POOLING_INDEXES[tup_i][sig_i]], 0)
                    for tup_i in range(len(outputs.last_hidden_state))
                    for sig_i in range(len(POOLING_INDEXES[tup_i]))])

        characteristic_vector = torch.stack([BASE_CHARACTERISTICS[tup_i][sig_i]
                    for tup_i in range(len(outputs.last_hidden_state))
                    for sig_i in range(len(POOLING_INDEXES[tup_i]))])

        term_score = self.final_layer(torch.cat((pooled_hidden_states, characteristic_vector), 1)).T

        return term_score

class LinkLlama(object):
    """The full model, containing both LLaMA and the prediction head."""
    def __init__(self, char_size, config, dataset):
        self.config = config

        self.curr_input_ids = None
        self.curr_signal_char_pool = None
        self.attribute_ids = None

        self.sample_buffer_size = self.config['buffer_size']
        self.sample_buffer_tokid = []
        self.sample_buffer_char = []
        self.sample_buffer_pool = []
        self.sample_buffer_attribute_ids = []
        self.sample_buffer_reinforcement = []

        self.buffer_sample_size = self.config['buffer_sample_size']

        self.linkyLlama = LinkLlamaModel(char_size, self.config, dataset).to(DEVICE)
        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.linkyLlama.parameters(), lr=config['learning_rate'])

    def _get_signal_characteristic_pairs(self, signals_of_interest, sender, tupID):
        signal_list = [splitSignal for signal in signals_of_interest for splitSignal in signal.splitByAttribute()]
        return (signal_list, [sender.featurize_term(signal, tupID) for signal in signal_list])

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
        self.curr_input_ids, self.curr_signal_char_pool, self.attribute_ids = \
            self.linkyLlama._prepare_sample(data, tupID,
                                           self._get_signal_characteristic_pairs(sender.signalIndex[tupID], sender, tupID))

        self.linkyLlama.eval()
        with torch.no_grad():
            results = self.linkyLlama(self.curr_input_ids.unsqueeze(0),
                                     self.curr_signal_char_pool[1].unsqueeze(0),
                                     [self.curr_signal_char_pool[2]],
                                     self.attribute_ids.unsqueeze(0))

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
                                    if len(self.sample_buffer_tokid) < self.buffer_sample_size
                                    else self.buffer_sample_size)
                                   )

        self.linkyLlama.train()
        self.optimizer.zero_grad()

        if split_sample:
            samples = [[x] for x in sample_idx]
        else:
            samples = [sample_idx]

        if self.config['lm_prefixes']:
            prefix_pre = self.linkyLlama.prefix_encoder.weight.data.detach().clone()
        attribute_pre = self.linkyLlama.attribute_encoder.weight.data.detach().clone()

        for samp in samples:
            if len(samp) == 0:
                continue

            pred = self.linkyLlama(
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_tokid[idx] for idx in samp], batch_first=True,
                                                padding_value=1),
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_char[idx] for idx in samp], batch_first=True),
                [self.sample_buffer_pool[idx] for idx in samp],
                torch.nn.utils.rnn.pad_sequence([self.sample_buffer_attribute_ids[idx] for idx in samp], batch_first=True))

            loss = self.loss_fn(pred, torch.tensor([self.sample_buffer_reinforcement[i][j] for i in samp for j in
                                                    range(len(self.sample_buffer_reinforcement[i]))],
                                                   dtype=torch.float32, device=DEVICE).unsqueeze(0)) / len(samples)

            loss.backward()  # call backwards on each one in order to acumulate gradients, then do a step to make the final backprop step.
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
            self.sample_buffer_attribute_ids.pop(0)
            self.sample_buffer_reinforcement.pop(0)

        return loss.item()
