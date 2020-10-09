import tensorflow as tf

import os
import time
import argparse

from model import *
from create_dataset import *



data_file_path = './cornell movie-dialogs corpus'
path_to_convs = os.path.join(data_file_path, 'movie_conversations.txt')
path_to_lines = os.path.join(data_file_path, 'movie_lines.txt')

root_logdir = os.path.join(os.curdir, "training_logs")


@tf.function
def train_step(m, r, encoder, decoder, tokenizer):
	loss = 0

	with tf.GradientTape() as tape:
		enc_out, hiddens = encoder(m)
		dec_in = (tf.expand_dims([tokenizer.word_index['<start>']] * r.shape[0], 1),
					hiddens, enc_out)

		for t in range(1, r.shape[1]):
			predictions, hiddens, _ = decoder(dec_in)
			loss += loss_function(r[:, t], predictions)
			dec_in = (tf.expand_dims(r[:, t], 1), hiddens, enc_out)

	batch_loss = loss / r.shape[1]

	variables = encoder.trainable_variables + decoder.trainable_variables

	gradients = tape.gradient(loss, variables)

	optimizer.apply_gradients(zip(gradients, variables))

	return batch_loss


@tf.function
def val_step(m, r, encoder, decoder, tokenizer):
	loss = 0
	enc_out, hiddens = encoder(m)
	dec_in = (tf.expand_dims([tokenizer.word_index['<start>']] * r.shape[0], 1),
				hiddens, enc_out)

	for t in range(1, r.shape[1]):
		predictions, hiddens, _ = decoder(dec_in)
		loss += loss_function(r[:, t], predictions)
		dec_in = (tf.expand_dims(r[:, t], 1), hiddens, enc_out)

	batch_loss = loss / int(r.shape[1])
	batch_perplexity = tf.exp(batch_loss)

	return batch_loss, batch_perplexity


# @tf.function
def summarize_and_save_epoch(epoch, start, train_loss, val_loss, val_perplexity,
	train_writer, val_writer, train_steps, val_steps):
	
	train_loss /= train_steps
	val_loss /= val_steps
	val_perplexity /= val_steps

	with train_writer.as_default():
		tf.summary.scalar('loss', train_loss, step=epoch)

	with val_writer.as_default():
		tf.summary.scalar('loss', val_loss, step=epoch)
		tf.summary.scalar('perplexity', val_perplexity, step=epoch)

	template = 'Epoch {} Train Loss {:.4f} Val Loss {:.4f} Val Perplexity {:.4f}'
	print(template.format(epoch, train_loss, val_loss, val_perplexity))
	print('Time taken for one epoch {} sec\n'.format(int(time.time() - start)))


def get_train_val_writers():
	run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
	
	train_logdir = os.path.join(root_logdir, run_id, 'train')
	val_logdir = os.path.join(root_logdir, run_id, 'val')

	train_writer = tf.summary.create_file_writer(train_logdir)
	val_writer = tf.summary.create_file_writer(val_logdir)

	return train_writer, val_writer


		
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training seq2seq model.')
	parser.add_argument('-epochs', '--epochs', type=int, metavar='', help='number of training epochs', default=20)

	parser.add_argument('-units', '--units', type=int, metavar='', help='number of neurons for GRU', default=512)
	parser.add_argument('-embedding_dim', '--embedding_dim', type=int, metavar='', help='embedding dimension', default=300)

	parser.add_argument('-vocab_size', '--vocab_size', type=int, metavar='', help='vocabulary size for the model', default=30000)
	parser.add_argument('-max_len', '--max_len', type=int, metavar='', help='max sentence length', default=45)
	parser.add_argument('-batch_size', '--batch_size', type=int, metavar='', help='batch size for the training set', default=64)
	parser.add_argument('-buffer_size', '--buffer_size', type=int, metavar='', help='buffer size for data shuffling', default=10000)
	parser.add_argument('-test_set_size', '--test_set_size', type=int, metavar='', help='size of the test set', default=10000)
	parser.add_argument('-val_batch_size', '--val_batch_size', type=int, metavar='', help='batch size for the validation set', default=100)
	
	parser.add_argument('-continue', '--continue_training', type=bool, metavar='', help='continue training the model from the last checkpoint', defualt=False)
	args = parser.parse_args()

	
	print('--> Creating encoder and decoder...\n')
	encoder = Encoder(int(args.units/2), args.embedding_dim, args.vocab_size)
	decoder = Decoder(args.units, encoder.embedding, args.vocab_size)
	#----------------------------------------------------------------
	
	print('--> Loading and preparing datasets...\n')
	message_train_val, response_train_val, tokenizer = load_dataset(
		path_to_convs, path_to_lines,
		args.max_len, args.vocab_size,
		args.test_set_size)

	m_train_tensor, m_val_tensor, m_val = message_train_val
	r_train_tensor, r_val_tensor, r_val = response_train_val

	train_set = tf.data.Dataset.from_tensor_slices((m_train_tensor, r_train_tensor))
	train_set = train_set.shuffle(args.buffer_size).batch(args.batch_size, drop_remainder=True)

	val_set = tf.data.Dataset.from_tensor_slices((m_val_tensor, r_val_tensor))
	val_set = val_set.shuffle(args.buffer_size).batch(args.val_batch_size, drop_remainder=True)


	steps_per_epoch = len(m_train_tensor) // args.batch_size
	steps_per_epoch_val = len(m_val_tensor) // args.val_batch_size
	#----------------------------------------------------------------

	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
	checkpoint = tf.train.Checkpoint(optimizer=optimizer,
		encoder=encoder,
		decoder=decoder)

	if args.continue_training:
		checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


	train_writer, val_writer = get_train_val_writers()

	print('--> Entering training loop...')
	print('--> Total Epochs:', args.epochs)


	# steps_per_epoch_val = steps_per_epoch = 2

	for epoch in range(1, args.epochs + 1):
		start = time.time()
		total_loss = 0
		for (batch, (m, r)) in enumerate(train_set.take(steps_per_epoch)):
			batch_loss = train_step(m, r, encoder, decoder, tokenizer)
			total_loss += batch_loss

			if batch % 100 == 0:
				print("Epoch {} Batch {}/{} Loss {:.4f}".format(epoch,batch,
					steps_per_epoch, batch_loss.numpy()))

		total_val_loss = 0
		total_val_perplexity = 0
		for (m, r) in val_set.take(steps_per_epoch_val):
			batch_val_loss, batch_perplexity = val_step(m, r, encoder, decoder, tokenizer)
			total_val_loss += batch_val_loss
			total_val_perplexity += batch_perplexity

		summarize_and_save_epoch(epoch, start, total_loss,total_val_loss, total_val_perplexity,
			train_writer, val_writer, steps_per_epoch, steps_per_epoch_val)

		if epoch % 2 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print("Epoch {}/{} Loss {:.4f}".format(epoch, args.epochs, total_loss/steps_per_epoch))
		print("Time taken for 1 eopch {} sec\n".format(int(time.time() - start)))
	






