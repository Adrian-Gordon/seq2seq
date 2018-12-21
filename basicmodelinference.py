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

#test_sequence_input = gd.true_signal(gd.train_data_x[-15:])
test_sequence_input, test_sequence_output = gd.getTrainingSample(1,config["input_sequence_length"], config["output_sequence_length"])
test_sequence_input = test_sequence_input[0]
test_sequence_output = test_sequence_output[0]
#print test_sequence_input
#print test_sequence_output

with tf.Session() as sess:
  sess.run(init)
  saver = tf.train.Saver
  saver().restore(sess, os.path.join('./',config["savefilename"]))
  feed_dict={encoder_inputs[t]: test_sequence_input[t].reshape(1,config["input_dim"]) for t in range(config["input_sequence_length"])}

  test_out = np.array(sess.run(seq2seqInference.encoder_decoder_inference,feed_dict)).reshape(-1)[:20]

  print(test_out)
 

  #l1, = plt.plot(range(85), gd.true_signal(gd.train_data_x[:85]), label = 'Training truth')
  #l2, = plt.plot(range(85, 105), gd.y[85:], 'yo', label = 'Test truth')
  #l3, = plt.plot(range(85,105), np.array(test_out).reshape(-1), 'r-', label = 'Test Predictions')
  l1, = plt.plot(range(config["input_sequence_length"]), test_sequence_input,'b-', label = 'Training input')
  l2, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_sequence_output,'c-', label = 'Training output')
  l3, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out,'ro', label = 'Predicted output')
 
  plt.legend(handles=[l1,l2,l3], loc = 'lower left')

  plt.show()