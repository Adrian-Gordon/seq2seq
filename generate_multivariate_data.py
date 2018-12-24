import tensorflow as tf
import numpy as np

import pandas as pd

class GenerateData:
  data =[]
  def __init__(self, datafile_path):
    GenerateData.data = pd.read_csv(datafile_path,names=['num','x','y1','y2','y3','y4','y5'])

  def getTrainingSample(self,batch_size, input_seq_length, output_seq_length):
    data = GenerateData.data
    input_batches = []
    output_batches = []
    n_starting_indexes = len(data) / (input_seq_length + output_seq_length)

    for i in range(batch_size):
      starting_index = np.random.randint(0,n_starting_indexes)
      starting_index_offset = starting_index * (input_seq_length + output_seq_length)

      an_input_batch_y = data[starting_index_offset: starting_index_offset + input_seq_length]
     # print(an_input_batch_y)
      input_data = an_input_batch_y[['y3','y4','y5']]
     # print(input_data)
      input_batches.append(np.array(input_data))

      an_output_batch_y = data[starting_index_offset + input_seq_length:starting_index_offset + input_seq_length + output_seq_length]
      output_data = an_output_batch_y[['y1','y2']]
      output_batches.append(np.array(output_data))
      #input_batches = np.array(input_batches).reshape(batch_size, input_seq_length, 3)
    #print(input_batches)
    #print(np.array(input_batches)).reshape(batch_size, input_seq_length, 3)
    return input_batches, output_batches


  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))
#test
#gd = GenerateData('multivariate-data.csv')

#input_batches, output_batches = gd.getTrainingSample(1,10,5)

#print input_batches
#print output_batches







 