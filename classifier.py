
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		# todo
		# Run the LLaMA model to obtain logits and hidden states for each position
		# Assumes self.llama returns (logits, hidden_states) where hidden_states is [B, T, H]
		logits, hidden_states = self.llama(input_ids)

		# Determine the index of the final (last non-PAD) token per sequence.
		# Prefer config.pad_token_id if available; otherwise, default to taking the last position.
		pad_token_id = getattr(self.llama.config, "pad_token_id", None)
		if pad_token_id is None:
			# No padding info available: take the final position directly
			final_indices = input_ids.new_full((input_ids.size(0),), input_ids.size(1) - 1)
		else:
			# Compute lengths = count of non-pad tokens per sequence
			lengths = (input_ids != pad_token_id).long().sum(dim=1)
			# Index of the final token is lengths - 1 (clamped to valid range)
			final_indices = torch.clamp(lengths - 1, min=0)

		# Gather the hidden state at the final token for each sequence: shape [B, H]
		batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
		final_hidden = hidden_states[batch_indices, final_indices, :]

		# Apply dropout
		final_hidden = self.dropout(final_hidden)

		# Classifier head to produce logits over classes: shape [B, num_labels]
		class_logits = self.classifier_head(final_hidden)

		# Return log-probabilities
		return F.log_softmax(class_logits, dim=-1)