from flask import Flask, render_template, request
from inference import Chatbot, load_model


app = Flask(__name__)
chatbot = load_model()


@app.route('/get_response')
def get_response_view():
	message = request.args.get('msg')
	return chatbot.beam_search(message)[1]


@app.route('/')
def home():
	return render_template('chatbot.html', title='Chatbot')


if __name__ == '__main__':
	app.run(debug=True)