import tensorflow as tf


class Encoder(tf.keras.Model):
	def __init__(self, enc_units, embedding_dim, vocab_size):
		super(Encoder, self).__init__()
		self.vocab_size = vocab_size

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

		self.bi_gru_1 = tf.keras.layers.Bidirectional(
			tf.keras.layers.GRU(enc_units,
				return_sequences=True,
				return_state=True,
				recurrent_initializer='glorot_uniform'))

		self.bi_gru_2 = tf.keras.layers.Bidirectional(
		    tf.keras.layers.GRU(enc_units,
		    	return_sequences=True,
		    	return_state=True,
		    	recurrent_initializer='glorot_uniform'))


	def call(self, enc_input):
		x = self.embedding(enc_input)
		sequence, fh_1, bh_1 = self.bi_gru_1(x)
		sequence, fh_2, bh_2 = self.bi_gru_2(sequence)
		hiddens = [tf.concat([fh_1, bh_1], axis=-1),
					tf.concat([fh_2, bh_2], axis=-1)]

		return sequence, hiddens


class LuongAttention(tf.keras.Model):
	def __init__(self, units):
		super(LuongAttention, self).__init__()
		self.W = tf.keras.layers.Dense(units)


	def call(self, attention_inputs):
		query, values = attention_inputs

		query = tf.expand_dims(query, axis=-1)
		values = self.W(values)

		scores = tf.matmul(values, query)

		attention_weights = tf.nn.softmax(scores, axis=1)

		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


class Decoder(tf.keras.Model):
	def __init__(self, dec_units, embedding_layer, vocab_size):
		super(Decoder, self).__init__()
		self.vocab_size = vocab_size

		self.embedding = embedding_layer

		self.gru_1 = tf.keras.layers.GRU(dec_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')
        
		self.gru_2 = tf.keras.layers.GRU(dec_units,
			return_sequences=True,
			return_state=True,
			recurrent_initializer='glorot_uniform')

		self.fc = tf.keras.layers.Dense(vocab_size, activation='softmax')
		
		self.attention = LuongAttention(dec_units)


	def call(self, dec_inputs):
		x, hiddens, enc_output = dec_inputs

		first_hidden, last_hidden = hiddens

		attention_inputs = (last_hidden, enc_output)

		context, attentions = self.attention(attention_inputs)

		x = self.embedding(x)
		x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

		output, first_state = self.gru_1(x, initial_state=first_hidden)

		output, last_state = self.gru_2(output, initial_state=last_hidden)
		
		output = tf.squeeze(output, axis=1)

		dec_output = self.fc(output)
		states = [first_state, last_state]

		return dec_output, states, attentions


optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
	from_logits=True, reduction='none')


def loss_function(real, pred):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_sum(loss_)

