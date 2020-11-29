import tensorflow as tf
import json
import re
from random import choice



MAX_LEN = 14


def load_tokenizer():
	with open('./tokenizer.json', 'r') as f:
		data = json.load(f)
		tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

	return tokenizer


class Chatbot:
	def __init__(self, encoder, decoder, tokenizer, max_len=MAX_LEN):
		self.encoder = encoder
		self.decoder = decoder
		self.tokenizer = tokenizer
		self.max_len = max_len

	def preprocess_text(self, text):
		text = re.sub(r"([?.!,¿])", r" \1 ", text)
		
		text = re.sub(r'[" "]+', " ", text)

		text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)

		text = text.strip()

		text = '<start> ' + text + ' <end>'
		
		return text

	def prepare_input(self, message):
		message = self.preprocess_text(message)

		sequence = self.tokenizer.texts_to_sequences([message])

		padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
			sequence,
			maxlen=self.max_len,
			truncating='post')

		tensor = tf.convert_to_tensor(padded_sequence)

		return tensor

	def greedy_search(self, message):
		tensor = self.prepare_input(message)

		response = ""

		enc_output, hiddens = self.encoder(tensor)

		dec_in = (tf.expand_dims([self.tokenizer.word_index['<start>']], 0),
			hiddens, enc_output)

		for t in range(self.max_len):
			pred, hiddens, _ = self.decoder(dec_in)

			pred_id = tf.argmax(pred[0]).numpy()

			pred_word = self.tokenizer.index_word[pred_id]

			if pred_word == '<end>':
				return message, response.strip()

			response += pred_word + ' '

			dec_in = (tf.expand_dims([pred_id], 0), hiddens, enc_output)

		return message, response.strip()

	def find_top_k(self, acc_val, ids, hiddens, enc_sequence, k):
		dec_in = (tf.expand_dims([ids[-1]], 0), hiddens, enc_sequence)
		pred, hiddens, _ = self.decoder(dec_in)

		top_k = tf.math.top_k(pred, k=k)
		top_vals = tf.nn.softmax(top_k.values).numpy()[0]
		top_indices = top_k.indices.numpy()[0]

		candidates = []

		for val, id_ in zip(top_vals, top_indices):
			candidates.append([val, ids + [id_], hiddens])

		return candidates

	def beam_search(self, message, k=5):
		start_token = self.tokenizer.word_index['<start>']
		end_token = self.tokenizer.word_index['<end>']

		tensor = self.prepare_input(message)

		enc_sequence, hiddens = self.encoder(tensor)

		candidates = self.find_top_k(1, [start_token], hiddens, enc_sequence, k)

		while True:
			next_candidates = []
			for candidate in candidates:
				if len(candidate[1]) == self.max_len or candidate[1][-1] == end_token:
					next_candidates.append(candidate)
					continue
				next_candidates.extend(self.find_top_k(*candidate, enc_sequence, k))

			candidates = sorted(next_candidates, reverse=True)[:k]

			are_ended = []
			for candidate in candidates:
				is_ended = len(candidate[1]) == self.max_len or candidate[1][-1] == end_token
				are_ended.append(is_ended)

			if all(are_ended):
				sequences = [cand[1][1:-1] for cand in candidates]
				# response = choice(self.tokenizer.sequences_to_texts(sequences))
				response = self.tokenizer.sequences_to_texts(sequences)[0]
				return message, response



if __name__ == '__main__':
	encoder_dir = './model/encoder_25ep'
	decoder_dir = './model/decoder_25ep'
	
	encoder = tf.keras.models.load_model(encoder_dir)
	decoder = tf.keras.models.load_model(decoder_dir)
	tokenizer = load_tokenizer()

	chatbot = Chatbot(encoder, decoder, tokenizer)

	while True:
		message = input('Me: ')
		print(f"Bot: {chatbot.beam_search(message)[1]}")
		# print(f"Bot: {chatbot.greedy_search(message)[1]}")







