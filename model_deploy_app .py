#importing libraries

import tensorflow as tf
from tensorflow import keras
from flask import Flask, request,  render_template
import numpy as np
import bert
import tensorflow_hub as hub
import re

def remove_url(sent):
	'''removes URLs from a string '''
	return re.sub(r"http\S+", "", sent)

def clean_sentence(sent):
	
	''' Remove unnecessary characters'''
	pattern = re.compile('[^\w\s]+|\n')
	s = pattern.sub('',sent)
	s = remove_url(s)
	s = pattern.sub('',s)
	return s

def padding(sent,maxlen = 160):
  
	'''Used for padding the sentence'''
	s = sent.split()
	no_of_words = len(s)
	if no_of_words<maxlen:
		s = sent + (maxlen-no_of_words)* ' [PAD]'
	elif no_of_words == maxlen:
		s = sent
	else:
    		raise Exception('Sentence length is more than maxlen')

	assert len(s.split()) == maxlen
	return s

'''using bert as a keras layer '''

bert_path = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(bert_path,name = "My_bert_layer") 

'''getting dictionary and tokenizer '''

dictionary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
bert_tok_hub = bert.bert_tokenization.FullTokenizer(dictionary_file, do_lower_case=True)

def tokenization(sent,maxlen = 160,tokenizer = bert_tok_hub):
  
	'''Add CLS and SEP and return tokenised sentence '''
	if maxlen<10:
		raise Exception('Maxlen should be more than 10')

  	#max_chars_betwn_cls_and_sep
	m = np.int((maxlen - 10)/2)
	len_of_tokens = len(tokenizer.tokenize(sent))
	s = '[CLS] ' + " ".join(tokenizer.tokenize(sent)[:m]) +' [SEP] '
	return s

def get_mask(sent,maxlen=160):

	'''Generating masks'''
	l = len(sent.split())
	if l>maxlen:
		raise Exception('Length is more than maxlen')
	mask = [1] * len(sent.split()) + [0] * (maxlen - len(sent.split()))
	return mask

def prep_data_for_ip(sent,maxlen = 160,tokenizer = bert_tok_hub):
  
	'''preparing data for input'''
	s = sent
	s = clean_sentence(sent)
	s_tokenized_string_with_cls_and_sep = tokenization(s,maxlen,tokenizer)
	s_padded = padding(s_tokenized_string_with_cls_and_sep,maxlen)
	s_m = get_mask(s_tokenized_string_with_cls_and_sep,maxlen)
	s_id = tokenizer.convert_tokens_to_ids(s_padded.split())
	s_seg = [0] * maxlen
	return [s_id,s_m,s_seg]


'''creating the flask app'''
app = Flask(__name__)
model = keras.models.load_model("model18/content/sample_data/model18")

statements = {0:'I can sense some informative content in the tweet',1:'There is some concerning aspect in the tweet',2:'I feel that is a casual tweet'}

@app.route('/')
def hello():
	return render_template("index.html")

@app.route('/h')
def hello1():
	return render_template("about.html")

@app.route('/hi')
def hello2():
	return render_template("h1.html")

@app.route('/result', methods=['POST'])
def result():
	fe =[ x for x in request.form.values()]
	print(fe)
	fe1 = fe[0]
	print(fe1)
	a,b,c= prep_data_for_ip(fe1)
	a = tf.cast(a,dtype = tf.int32)
	b = tf.cast(b,dtype = tf.int32)
	c = tf.cast(c,dtype = tf.int32)
	a = tf.expand_dims(a,0)
	b = tf.expand_dims(b,0)
	c = tf.expand_dims(c,0)
	pred = model.predict([a,b,c])
	pred_class = np.argmax(pred)
	sentiment = statements[pred_class]
	return render_template('h1.html', text="{}".format(sentiment))


if __name__ == "__main__":
	app.run(debug=True)