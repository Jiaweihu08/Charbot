import tensorflow as tf
import json
import argparse

from model import *
from create_dataset import unicoder_to_ascii, preprocess_text

max_len = 18

def load_model(units, embedding_dim, vocab_size, checkpoint_dir='./training_checkpoints',
	tokenizer_dir='./tokenizer.json'):
	encoder = Encoder(units, embedding_dim, vocab_size)
	decoder = Decoder(units, encoder.embedding, vocab_size)

	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
		encoder=encoder,
		decoder=decoder)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

	with open(tokenizer_dir) as f:
		data = json.load(f)
		tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

	return encoder, decoder, tokenizer


def get_input(message):
	message = preprocess_text(message)

	inputs = [tokenizer.word_index.get(w, 0) for w in message.split()]
	inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
		maxlen=max_len, truncating='post')

	inputs = tf.convert_to_tensor(inputs)

	return inputs


def get_response(message, encoder, decoder, tokenizer):
	results = ''

	enc_out, hiddens = encoder(inputs)
	dec_in = (tf.expand_dims([tokenizer.word_index['<start>']], 0), hiddens, enc_out)

	for t in range(max_len):
		predictions, hiddens = decoder(dec_in)
		prediction_id = tf.argmax(predictions[0]).numpy()
		results += tokenizer.index_word[prediction_id] + ' '
		if tokenizer.index_word[prediction_id] == '<end>':
			return message, results.strip()

		dec_in = (tf.expand_dims([prediction_id], 0), hiddens, enc_out)

	return message, results.strip()


def beam_search(message, encoder, decoder, tokenizer):
	pass


if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description='Inferencing model')
	# parser.add_argument('-units', '--units', type=int, metavar='', help='number of neurons for GRU', default=512)
	# parser.add_argument('-embedding_dim', '--embedding_dim', type=int, metavar='', help='embedding dimension', default=300)
	# parser.add_argument('-vocab_size', '--vocab_size', type=int, metavar='', help='vocabulary size for the model', default=30000)
	# args = parser.parse_args()

	# encoder, decoder, tokenizer = load_model(args.units, args.embedding_dim, args.vocab_size)
	# message = os.stdin.read()
	# _, response = get_response(message, encoder, decoder, tokenizer)
	# print(message)
	# print(response)
	





