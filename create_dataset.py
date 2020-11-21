import tensorflow as tf

import os
import re
import unicodedata
import json
from sklearn.model_selection import train_test_split


# only consider words that appeared more than twice
VOCAB_SIZE = 9561
MAX_LEN = 12
BATCH_SIZE = 32
TEST_BATCH_SIZE = 100
TEST_SET_SIZE = 2000


def unicode_to_ascii(s):
	"""Source:
	https://www.tensorflow.org/tutorials/text/nmt_with_attention
	"""
	return ''.join(c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn')


def preprocess_text(text):
	"""Source: https://www.tensorflow.org/tutorials/text/nmt_with_attention
	"""
	text = unicode_to_ascii(text.lower().strip())

	# creating a space between a word and the punctuation following it
	# eg: "he is a boy." => "he is a boy ."
	# Reference:- https://stackoverflow.com/questions/3645931/
	# python-padding-punctuation-with-white-spaces-keeping-punctuation

	text = re.sub(r"([?.!,¿])", r" \1 ", text)
	text = re.sub(r'[" "]+', " ", text)

	# replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
	text = re.sub(r"[^a-zA-Z0-9?.!,]+", " ", text)

	text = text.strip()

	# adding a start and an end token to the sentence
	# so that the model know when to start and stop predicting.
	text = '<start> ' + text + ' <end>'
	return text


def load_source_data(path_to_convs, path_to_lines):
	with open(path_to_convs, encoding='iso-8859-1') as f:
		convs = f.read().strip().split('\n')

	convs = [re.findall(r'L\d+', conv) for conv in convs]
    
	with open(path_to_lines, encoding='iso-8859-1') as f:
		lines = f.read()strip().split('\n')
    
	line_dict = dict()
	for line in lines:
		line = line.split(' +++$+++ ')
		text = preprocess_text(line[-1])

		line_dict[line[0]] = text

	return convs, line_dict


def create_conv_pairs(path_to_convs, path_to_lines, max_len=MAX_LEN):
	"""Create training instances, message as input and response and
	correct predictions.
	For each conversation in the form of [l1, l2, l3, ..., ln],
	create pairs of instances in the form of:
	[l1, l2], [l2, l3], ..., [l(n-1), ln]
	"""
	convs, line_dict = load_source_data(path_to_convs, path_to_lines)

	messages = []
	responses = []
	for turns in convs:
		for i in range(len(turns)-1):
			m = line_dict[turns[i]]
			r = line_dict[turns[i+1]]
			if len(m.split()) > max_len or len(r.split()) > max_len:
				continue
			messages.append(m)
			responses.append(r)

	return messages, responses


def save_to_train_test_files(
	path_to_convs, path_to_lines, path_to_train_set, path_to_test_set):
	
	messages, responses = create_conv_pairs(path_to_convs, path_to_lines)

	assert len(messages) == len(responses)

	print(f"number of total instances: {len(messages)}")

	m_train, m_test, r_train, r_test = train_test_split(
		messages, responses, test_size=TEST_SET_SIZE, random_state=42)

	print(f"number of training instances: {len(m_train)}")
	print(f"number of test instances: {len(m_test)}")
	print("Saving data to files...")

	breaker = " +++ "

	with open(path_to_train_set, 'w') as f:
		for i in range(len(m_train)):
			f.write(m_train[i] + breaker + r_train[i] + '\n')

	with open(path_to_test_set, 'w') as f:
		for i in range(len(m_test)):
			f.write(m_test[i] + breaker + r_test[i] + '\n')


def load_dataset(path_to_dataset):
	with open(path_to_dataset) as f:
		lines = f.read().strip().split('\n')

	messages = []
	responses = []
	breaker = ' +++ '
	for line in lines:
		m, r = line.split(breaker)
		messages.append(m)
		responses.append(r)

	print(f'- number of instances: {len(messages)}')

	return messages, responses


def get_tensor(message, response, tokenizer, max_len):
	m_tensor = tokenizer.texts_to_sequences(message)
	m_tensor = tf.keras.preprocessing.sequence.pad_sequences(
		m_tensor,
		maxlen=max_len,
		padding='post',
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
	tokenizer_json = tokenizer.to_json()
	with open(file_name, 'w', encoding='utf-8') as f:
		f.write(json.dumps(tokenizer_json, ensure_ascii=False))
		print('Tokenizer saved at %s' % file_name)


def get_dataset(path_to_dataset, vocab_size, max_len,
	batch_size, tokenizer=None):
	
	messages, responses = load_dataset(path_to_dataset)
	steps_per_epoch = len(messages) // batch_size

	if tokenizer == None:
		tokenizer = tf.keras.preprocessing.text.Tokenizer(
			num_words=vocab_size,
			filters='')
		
		tokenizer.fit_on_texts(messages + responses)

		save_tokenizer(tokenizer)

	message_tensor, response_tensor = get_tensors(
		messages, responses,
		tokenizer, max_len)

	print(f'- tensor shape: {message_tensor.shape}')
	print(f'- steps per spoch: {steps_per_epoch}')

	dataset = tf.data.Dataset.from_tensor_slices(
		(message_tensor, response_tensor))
	dataset = dataset.shuffle(buffer_size=100000).batch(
		batch_size, drop_remainder=True)

	return dataset, tokenizer, steps_per_epoch



if __name__ == '__main__':
	# import argparse

	# parser = argparse.ArgumentParser("Loading source data, preprocess the data, create and save training and testing data\
	# 	to their files.")
	# parser.add_argument('-vocab_size', '--vocab_size', type=int, metavar='', help='vocabulary size for the model', default=9668)
	# parser.add_argument('-max_len', '--max_len', type=int, metavar='', help='max sentence length', default=12)
	# parser.add_argument('-test_set_size', '--test_set_size', type=int, metavar='', help='size of the test set', default=5000)

	# args = parser.parse_args()

	data_file_path = './cornell movie-dialogs corpus'
	path_to_convs = os.path.join(data_file_path, 'movie_conversations.txt')
	path_to_lines = os.path.join(data_file_path, 'movie_lines.txt')

	path_to_train_set = './train_test_data/train_set.txt'
	path_to_test_set = './test_test_data/test_set.txt'
	
	save_to_train_test_files(path_to_convs, path_to_lines, path_to_train_set, path_to_test_set)




