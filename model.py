import tensorflow as tf


class Encoder(tf.keras.Model):
	def __init__(self, enc_units, embedding_dim, vocab_size):
		super(Encoder, self).__init__()
		self.vocab_size = vocab_size

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

		self.gru_1 = tf.keras.layers.GRU(
			enc_units,
			return_sequences=True,
			return_state=True,
			dropout=0.1,
			recurrent_initializer='glorot_uniform')

		self.gru_2 = tf.keras.layers.GRU(
			enc_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')


	def call(self, enc_input):
		# enc_input shape: (batch_size, max_len)
		# x_emb shape: (batch_size, max_len, units)
		x_emb = self.embedding(enc_input)

		# sequence shape: (bvatch_size, max_len, units)
		# hiddens shape: (batch_size, units)
		sequence_1, hiddens_1 = self.gru_1(x_emb)
		sequence_2, hiddens_2 = self.gru_2(sequence_1)

		hiddens = [hiddens_1, hiddens_2]

		return sequence_2, hiddens


class LuongAttention(tf.keras.Model):
	def __init__(self, units):
		super(LuongAttention, self).__init__()
		self.W = tf.keras.layers.Dense(units)


	def call(self, attention_inputs):
		""" Luong's Attention uses the current decoder's output
		as query to find attention weights. The obtained context vector
		is used directly to compute the predictions.
		"""
		# query shape: (batch_size, 1, units)
		# values shape: (batch_size, max_len, units)
		query, values = attention_inputs

		# scores shape: (batch_size, 1, max_len)
		scores = tf.matmul(query, self.W(values), transpose_b=True)

		# alignments/attention weights shape: (batch_size, 1, max_len)
		# it measures the similarity between the query and the values
		alignments = tf.nn.softmax(scores, axis=-1)

		# context_vecrtor shape: (batch_size, 1, units)
		# the obtained alignments is used to compute the most important
		# components from the values to compute the predictions
		context_vector = tf.matmul(alignments, values)

		return context_vector, alignments


class Decoder(tf.keras.Model):
	def __init__(self, dec_units, embedding_layer, vocab_size):
		super(Decoder, self).__init__()
		# unlike the case in Neural Machine Translation, the encoder and
		# the decoder here share the same embedding layer, for that, we take the
		# embedding layer from the encoder and use it directly as the embeddings
		# of the decoder
		self.embedding = embedding_layer

		self.gru_1 = tf.keras.layers.GRU(
			dec_units,
			return_sequences=True,
			return_state=True,
			dropout=0.1,
			recurrent_initializer='glorot_uniform')

		self.gru_2 = tf.keras.layers.GRU(
			dec_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')

		self.attention = LuongAttention(dec_units)

		self.fc = tf.keras.layers.Dense(dec_units, activation='tanh')

		self.out = tf.keras.layers.Dense(vocab_size)


	def call(self, dec_inputs):
		# x shape: (batch_size, 1)
		# hiddens shape: (batch_size, units)
		# enc_outputs shape: (batch_size, max_len, units)
		x, hiddens, enc_output = dec_inputs
		input_hiddens_1, input_hiddens_2 = hiddens

		# x_emb shape: (batch_size, max_len, embedding_dim)
		x_emb = self.embedding(x)

		# sequence_2 is used as query for the attention model
		# sequence shape: (batch_size, 1, units)
		# hiddens shape: (batch_size, units)
		sequence_1, hiddens_1 = self.gru_1(x_emb, initial_state=input_hiddens_1)
		sequence_2, hiddens_2 = self.gru_2(sequence_1, initial_state=input_hiddens_2)
		
		hiddens = [hiddens_1, hiddens_2]

		# context vector shape: (batch_size, 1, units)
		# alignments shape: (batch_size, 1, max_len)
		attention_inputs = (sequence_2, enc_output)
		context_vector, alignments = self.attention(attention_inputs)

		# output_sequence shape after squeezing: (batch_size, 2 * units)
		output_sequence = tf.concat([context_vector, sequence_2], axis=-1)
		output_sequence = tf.squeeze(output_sequence, axis=1)

		# use the fc layer to shape the sequence back to
		# shape: (batch_size, units)
		output_sequence = self.fc(output_sequence)

		# final predictions, shape: (batch_size, vocab_size)
		# outputting predictions for each component from the batch
		logits = self.out(output_sequence)

		return logits, hiddens, alignments


optimizer = tf.keras.optimizers.Adam(0.0001)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=True, reduction='none')


def loss_function(real, pred):
	"""Computing loss at each iteration
	The function first computes a mask for the padding components
	that has an id of 0, the mask if later used to filter the components
	that shouldm't be taken into account.
	"""
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)

