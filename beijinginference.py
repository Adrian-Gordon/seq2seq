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


sequence_length = config["input_sequence_length"] + config["output_sequence_length"]

n_iterations = (len(GenerateData.X_test) / config["output_sequence_length"]) -6

#test_sequence_output = test_sequence_output[0]
#print(sequence_length, n_iterations)

with tf.Session() as sess:
  sess.run(init)
  saver = tf.train.Saver
  saver().restore(sess, os.path.join('./',config["savefilename"]))

  offset = 0
  actual_output=[]
  test_output=[]

  for iteration in range(n_iterations):
    #print iteration
    test_sequence_input, test_sequence_output = gd.getTestSample(config["input_sequence_length"], config["output_sequence_length"],offset)
    test_sequence_input = test_sequence_input[0]
    feed_dict={encoder_inputs[t]: test_sequence_input[t].reshape(1,config["input_dim"]) for t in range(config["input_sequence_length"])}
    feed_dict.update({decoder_target_inputs[t]: np.zeros([1,config["output_dim"]]) for t in range(config["output_sequence_length"])})

    test_out = np.array(sess.run(seq2seqInference.encoder_decoder_inference,feed_dict)).transpose()#.reshape(-1)[:20]
    test_in = test_sequence_input.transpose()
    actual_output.append(test_sequence_output.reshape(-1))
    test_output.append(test_out)
    offset+=config["output_sequence_length"]

    #print("tsi: ", test_sequence_input)
    #print("tso: ", test_sequence_output)

    #print("test_out: ", test_out)
  actual_output=np.array(actual_output).reshape(-1)
  test_output=np.array(test_output).reshape(-1)
  print("Test mse is: ", np.mean((actual_output - test_output)**2))
#Univariate case
 # l1, =plt.plot(range(config["input_sequence_length"]),test_in,'c-', label = 'Training input')
 # l2, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out.reshape(-1),'ro', label = 'Predicted output')
 
 # plt.legend(handles=[l1,l2], loc = 'lower left')

 # plt.show()

#multivariate case
#  l1, = plt.plot(range(config["input_sequence_length"]),test_in[0] ,'b-', label = 'Sample input 0')
#  l2, = plt.plot(range(config["input_sequence_length"]),test_in[1] ,'c-', label = 'Sample input 1')
#  l3, = plt.plot(range(config["input_sequence_length"]),test_in[2] ,'m-', label = 'Sample input 1')
#  l3, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out[0][0],'ro', label = 'Predicted output 0')
#  l4, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out[1][0],'bo', label = 'Predicted output 1')
#  plt.legend(handles=[l1,l3, l4], loc = 'lower left')

#beijing case
#l3, =plt.plot(range(config["input_sequence_length"],config["input_sequence_length"] + config["output_sequence_length"]),test_out[0][0],'r-', label = 'Predicted output')
l3, =plt.plot(actual_output,'b-', label = 'Actual output ')
l4, =plt.plot(test_output, 'r-', label = 'Test output')
plt.legend(handles=[l3,l4], loc = 'lower left')


plt.show()
 