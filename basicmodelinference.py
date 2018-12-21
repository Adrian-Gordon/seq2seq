#basic seqtoseq inference model for time series
#usage: python basicmodelinference.py <configfilename>

import tensorflow as tf

import os

import sys

import json

from matplotlib import pyplot as plt


from basicmodel import *

configfilename = sys.argv[1]

with open(configfilename,'r') as f:
  config = json.load(f)

generate_data_library = config["data_modulename"]

cmd = 'from ' + generate_data_library + ' import *'

exec(cmd)

 #now do some inference

#create a seq2seq instance
seq2seqInference  = Seq2Seq()
init = tf.global_variables_initializer()

gd=GenerateData(config["datafilename"])


test_sequence_input, test_sequence_output = gd.getTrainingSample(1,config["input_sequence_length"], config["output_sequence_length"])
test_sequence_input = test_sequence_input[0]
test_sequence_output = test_sequence_output[0]


with tf.Session() as sess:
  sess.run(init)
  saver = tf.train.Saver
  saver().restore(sess, os.path.join('./',config["savefilename"]))
  feed_dict={encoder_inputs[t]: test_sequence_input[t].reshape(1,config["input_dim"]) for t in range(config["input_sequence_length"])}
  feed_dict.update({decoder_target_inputs[t]: np.zeros([1,config["output_dim"]]) for t in range(config["output_sequence_length"])})

  test_out = np.array(sess.run(seq2seqInference.encoder_decoder_inference,feed_dict)).transpose()#.reshape(-1)[:20]
  test_in = test_sequence_input.transpose()
 
#Univariate case
 # l1, =plt.plot(range(config["input_sequence_length"]),test_in,'c-', label = 'Training input')
 # l2, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out.reshape(-1),'ro', label = 'Predicted output')
 
 # plt.legend(handles=[l1,l2], loc = 'lower left')

 # plt.show()

#multivariate case
  l1, = plt.plot(range(config["input_sequence_length"]),test_in[0] ,'b-', label = 'Sample input 0')
  l2, = plt.plot(range(config["input_sequence_length"]),test_in[1] ,'c-', label = 'Sample input 1')
  l3, = plt.plot(range(config["input_sequence_length"]),test_in[2] ,'m-', label = 'Sample input 1')
  l3, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out[0][0],'ro', label = 'Predicted output 0')
  l4, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out[1][0],'bo', label = 'Predicted output 1')
  plt.legend(handles=[l1,l3, l4], loc = 'lower left')

  plt.show()
 