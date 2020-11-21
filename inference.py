import tensorflow as tf
import json
import unicodedata

from model import *


MAX_LEN = 12

def load_model(units, embedding_dim, vocab_size,
		checkpoint_dir='./training_checkpoints',
		tok_path='./tokenizer.json'):
	
	encoder = Encoder(units, embedding_dim, vocab_size)
	decoder = Decoder(units, encoder.embedding, vocab_size)

	checkpoint = tf.train.Checkpoint(
		optimizer=optimizer,
		encoder=encoder,
		decoder=decoder)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

	with open(tokenizer_dir) as f:
		data = json.load(f)
		tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

	return encoder, decoder, tokenizer


class Chatbot:
	"""class for greedy output generation"""
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
		message = preprocess_input(message)

		sequence = self.tokenizer.texts_to_sequences([message])

		padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
			sequence,
			maxlen=self.max_len
			truncating='post')

		tensor = tf.convert_to_tensor(padded_sequence)

		return tensor

	def __call__(self, message):
		tensor = self.prepare_input(message)

		response = ""

		enc_output, hiddens = self.encoder(tensor)

		dec_in = (tf.expand_dims([self.tokenizer.word_index['<start>']], 0),
			hiddens, enc_outpput)

		for t in range(self.max_len):
			pred, hiddens, _ = self.decoder(dec_in)

			pred_id = tf.argmax(pred[0]).numpy()

			pred_word = self.tokenizer.index_word[pred_id]

			if pred_word == '<end>':
				return message, response.strip()

			response += pred_word + ' '

			dec_in = (tf.expand_dims([pred_id], 0), hiddens, enc_output)

		return message, response.strip()


