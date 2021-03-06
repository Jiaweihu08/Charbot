# Chitchatting Chatbo - A seq2seq model with attention.

![](static/demo_gif.gif)
### Motivation
	- Gain deepened understanding on NLP
	- Hands-on experience implementing seq2seq models with attention with TensorFlow
	- Familiarize model training using TPUs

### Steps
	- Theoretical understanding on Encoder-Decoder based seq2seq conversational agent with attentions
	- Explore and preprocess a variety of conversational datasets
	- Build encoder-decoder model with attention:
		1. Shared embedding layer for encoder and decoder
		2. Two GRU layer for both encoder and decoder
		3. Bidirectional encoder
		3. Decoder with Luong's attention
	- Define training loop with teacher-forcing
	- Inference with both greedy and beam search
	- Deploy the model with Flask

### How to use:
	- Clone the repo
	- Create an virtual environment and install the packeges using: pip install -r requirements.txt
	- run flask app: python app.py
