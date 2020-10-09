import tensorflow as tf

import argparse
import json
from nltk.translate.bleu_score import corpus_bleu, SmoothFunction
from model import *


def calculate_corpus_bleu(encoder, decoder, tokenizer, m_val_tensor, r_val_mod):
	batch_size = 1000
	steps = len(m_val_tensor) // batch_size
	all_responses = []

	for i in range(steps):
		print("Step {}/{}".format(i + 1, steps))
		
		begin = time.time()
		start = i * batch_size
		end = (i + 1) * batch_size

		m_val_batch = m_val_tensor[start:end]
		r_val_mod_batch = r_val_mod[start:end]

		enc_out, hiddens = encoder(m_val_batch)
		dec_in = tf.expand_dims([tokenizer.word_index['<start>']] * batch_size, 1)

		batch_responses = []
		for i in range(MAX_SENT_LEN):
			predictions, hiddens, _ = decoder(dec_in, hiddens, enc_out)
			prediction_ids = tf.argmax(predictions, axis=1)

			ids = list(map(lambda id: tokenizer.index_word.get(id, '<UNK>'),
				prediction_ids.numpy()))

			batch_responses.append(ids)

			dec_in = tf.expand_dims(prediction_ids, 1)

		generated_outputs = list(zip(*batch_responses))

		for i in range(batch_size):
			for j in range(MAX_SENT_LEN):
				if generated_outputs[i][j] == '<end>':
					generated[i] = generated[i][:j]
					break

		all_responses.extend(generated)
		print("Time taken for 1 epoch {} secs\n".format(time.time()-begin))

	smooth_func = SmoothFunction()
	bleu = corpus_bleu(r_val_mod, all_responses, smooth_function=smooth_func)

	return bleu



# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser(description='Get BLEU score for the model')
# 	parser.add_argument('-units', '--units', type=int, metavar='', help='number of neurons for GRU', default=512)
# 	parser.add_argument('-embedding_dim', '--embedding_dim', type=int, metavar='', help='embedding dimension', default=300)
# 	parser.add_argument('-vocab_size', '--vocab_size', type=int, metavar='', help='vocabulary size for the model', default=30000)
# 	args = parser.parse_args()


# 	encoder = Encoder(int(args.units/2), args.embedding_dim, args.vocab_size)
# 	decoder = Decoder(args.units, encoder.embedding, args.vocab_size)

# 	checkpoint_dir = './training_checkpoints'
# 	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
# 	checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
# 	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# 	with open('tokniezer.json') as f:
# 		data = json.load(f)
# 		tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)


# 	message_train_val, response_train_val, tokenizer = load_dataset(
# 															path_to_convs, path_to_lines,
# 															args.max_len, args.vocab_size,
# 															args.test_set_size)

# 	m_train_tensor, m_val_tensor, m_val = message_train_val
# 	r_train_tensor, r_val_tensor, r_val = response_train_val