import tensorflow as tf


class Encoder(tf.keras.Model):
	def __init__(self, enc_units, embedding_dim, vocab_size):
		super(Encoder, self).__init__()
		self.vocab_size = vocab_size

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

		self.gru = tf.keras.layers.GRU(enc_units,
			return_sequences=True,
			return_state=True,
			dropout=0.25,
			recurrent_initializer='glorot_uniform')


	def call(self, enc_input):
		x_emb = self.embedding(enc_input)
		
		sequence, hiddens = self.gru(x_emb)

		return sequence, hiddens


class LuongAttention(tf.keras.Model):
	def __init__(self, units):
		super(LuongAttention, self).__init__()
		self.W = tf.keras.layers.Dense(units)


	def call(self, attention_inputs):
		query, values = attention_inputs

		values = self.W(values)

		scores = tf.matmul(query, values, transpose_b=True)

		alignments = tf.nn.softmax(scores, axis=-1)

		context_vector = tf.matmul(alignments, values)

		return context_vector, alignments


class Decoder(tf.keras.Model):
	def __init__(self, dec_units, embedding_layer, vocab_size):
		super(Decoder, self).__init__()

		self.embedding = embedding_layer

		self.gru = tf.keras.layers.GRU(dec_units,
			return_sequences=True,
			return_state=True,
			dropout=0.25,
			recurrent_initializer='glorot_uniform')

		self.attention = LuongAttention(dec_units)

		self.fc = tf.keras.layers.Dense(dec_units, activation='tanh')

		self.out = tf.keras.layers.Dense(vocab_size, activation='softmax')


	def call(self, dec_inputs):
		x, hiddens, enc_output = dec_inputs

		x_emb = self.embedding(x)

		gru_out, hiddens = self.gru(x_emb, initial_state=hiddens)

		attention_inputs = (gru_out, enc_output)
		context, alignments = self.attention(attention_inputs)

		gru_out = tf.concat([gru_out, context], axis=-1)
		gru_out = tf.squeeze(gru_out, axis=1)

		gru_out = self.fc(gru_out)

		logits = self.out(gru_out)

		return logits, hiddens, alignments


optimizer = tf.keras.optimizers.Adam(0.0001)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=True, reduction='none')


def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_sum(loss_)

