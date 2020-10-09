import tensorflow as tf

import os
import re
import unicodedata
import json
from sklearn.model_selection import train_test_split



def unicode_to_ascii(s):
	"""Source:
	https://www.tensorflow.org/tutorials/text/nmt_with_attention
	"""
	return ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')


def preprocess_text(w):
	"""Source:
	https://www.tensorflow.org/tutorials/text/nmt_with_attention
	"""
	w = unicode_to_ascii(w.lower().strip())

	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	# Reference:- https://stackoverflow.com/questions/3645931/
	# python-padding-punctuation-with-white-spaces-keeping-punctuation

	w = re.sub(r"([?.!,¿])", r" \1 ", w)
	w = re.sub(r'[" "]+', " ", w)

	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	w = re.sub(r"[^a-zA-Z?.!,'¿]+", " ", w)

	w = w.strip()

	# adding a start and an end token to the sentence
	# so that the model know when to start and stop predicting.
	w = '<start> ' + w + ' <end>'
	return w


def load_data(path_to_convs, path_to_lines):
	with open(path_to_convs) as f:
		convs = f.read().strip().split('\n')

	convs = [re.findall(r'L\d+', conv) for conv in convs]
    
	with open(path_to_lines, 'rb') as f:
		lines = f.read().decode('utf-8', 'ignore')
		lines = lines.strip().split('\n')
    
	line_dict = dict()
	for line in lines:
		line = line.split(' +++$+++ ')
		text = preprocess_text(line[-1])

		line_dict[line[0]] = text

	return convs, line_dict


def create_dataset(path_to_convs, path_to_lines):
	convs, line_dict = load_data(path_to_convs, path_to_lines)

	conv_pairs = []
	for turns in convs:
		for i in range(len(turns)-1):
			conv_pairs.append([line_dict[turns[i]], line_dict[turns[i+1]]])

	return zip(*conv_pairs), list(line_dict.values())


def get_tensor(message, response, tokenizer, max_len):
	m_tensor = tokenizer.texts_to_sequences(message)
	m_tensor = tf.keras.preprocessing.sequence.pad_sequences(
		m_tensor,
		maxlen=max_len,
		truncating='post')

	r_tensor = tokenizer.texts_to_sequences(response)
	r_tensor = tf.keras.preprocessing.sequence.pad_sequences(
		r_tensor,
		maxlen=max_len,
		padding='post',
		truncating='post')

	return m_tensor, r_tensor


def save_tokenizer(tokenizer):
	file_name = './tokenizer.json'
	if not os.path.isfile(file_name):
		tokenizer_json = tokenizer.to_json()
		with open(file_name, 'w', encoding='utf-8') as f:
			f.write(json.dumps(tokenizer_json, ensure_ascii=False))
			print('Tokenizer saved at %s' % file_name)


def load_dataset(path_to_convs, path_to_lines, max_len, vocab_size, test_set_size):

	data, lines = create_dataset(path_to_convs, path_to_lines)
	messages, responses = data

	tokenizer = tf.keras.preprocessing.text.Tokenizer(
		num_words=vocab_size,
		filters='')
	tokenizer.fit_on_texts(lines)

	save_tokenizer(tokenizer)

	m_train, m_val, r_train, r_val = train_test_split(
		messages, responses,
		test_size=test_set_size,
		random_state=42)

	m_train_tensor, r_train_tensor = get_tensor(m_train, r_train, tokenizer, max_len)

	m_val_tensor, r_val_tensor = get_tensor(m_val, r_val, tokenizer, max_len)


	return (m_train_tensor, m_val_tensor, m_val), (r_train_tensor, r_val_tensor, r_val), tokenizer


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser("Testing util functions for dataset creation")
	parser.add_argument('-vocab_size', '--vocab_size', type=int, metavar='', help='vocabulary size for the model', default=30000)
	parser.add_argument('-max_len', '--max_len', type=int, metavar='', help='max sentence length', default=45)
	parser.add_argument('-test_set_size', '--test_set_size', type=int, metavar='', help='size of the test set', default=10000)

	args = parser.parse_args()

	data_file_path = './cornell movie-dialogs corpus'
	path_to_convs = os.path.join(data_file_path, 'movie_conversations.txt')
	path_to_lines = os.path.join(data_file_path, 'movie_lines.txt')

	print('--> Loading and preparing datasets...')
	message_train_val, response_train_val, tokenizer = load_dataset(
														path_to_convs, path_to_lines,
														args.max_len, args.vocab_size,
														args.test_set_size)

	m_train_tensor, m_val_tensor, m_val = message_train_val
	r_train_tensor, r_val_tensor, r_val = response_train_val

	print("training tensor shape", m_train_tensor.shape)
	print("testing tensor shape", m_val_tensor.shape)


