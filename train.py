import tensorflow as tf

import os
import time

from model import *
from create_dataset import get_dataset


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


def summarize_and_save_epoch(epoch, start, train_loss,
		train_writer, train_steps):
	
	train_loss /= train_steps

	with train_writer.as_default():
		tf.summary.scalar('loss', train_loss, step=epoch)

	print('Epoch {} Train Loss {:.4f}'.format(epoch, train_loss))
	print('Time taken for one epoch {} sec\n'.format(int(time.time() - start)))


def get_train_writer():
	run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
	
	train_logdir = os.path.join(root_logdir, run_id, 'train')

	train_writer = tf.summary.create_file_writer(train_logdir)

	return train_writer



if __name__ == '__main__':
	EPOCHS = 4000
	units = 1024
	embedding_dim = 512
	VOCAB_SIZE = 9561

	path_to_train_set = './train_test_data/train_set.txt'
	root_logdir = os.path.join(os.curdir, "training_logs")
	
	#----------------------------------------------------------------	
	print('--> Loading the training set...\n')
	train_set, tokenizer, steps_per_epoch = get_dataset(path_to_train_set)

	#----------------------------------------------------------------
	print('--> Creating encoder and decoder...\n')
	encoder = Encoder(units, embedding_dim, VOCAB_SIZE)
	decoder = Decoder(units, encoder.embedding, VOCAB_SIZE)

	checkpoint_dir = './training_checkpoints'
	checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
	checkpoint = tf.train.Checkpoint(
		optimizer=optimizer,
		encoder=encoder,
		decoder=decoder)

	#----------------------------------------------------------------
	train_writer = get_train_writer()

	#----------------------------------------------------------------
	print('--> Entering training loop...')
	print('--> Total Epochs:', EPOCHS)

	for epoch in range(EPOCHS):
		start = time.time()
		total_loss = 0
		for (batch, (m, r)) in enumerate(train_set.take(steps_per_epoch)):
			batch_loss = train_step(m, r, encoder, decoder, tokenizer)
			total_loss += batch_loss

			if batch % 1 == 0:
				print("Epoch {} Batch {}/{} Loss {:.4f}".format(
					epoch, batch, steps_per_epoch, batch_loss.numpy()))


		summarize_and_save_epoch(epoch, start, total_loss,
			train_writer, steps_per_epoch)

		if epoch % 2 == 0:
			checkpoint.save(file_prefix=checkpoint_prefix)

		print("Epoch {}/{} Loss {:.4f}".format(epoch, EPOCHS, total_loss/steps_per_epoch))
		print("Time taken for 1 eopch {} sec\n".format(int(time.time() - start)))
	






