
import tensorflow as tf
import numpy as np

import pandas as pd

class GenerateData:
  data =[]
  def __init__(self, datafile_path):
    GenerateData.data = pd.read_csv(datafile_path,names=['num','x','y'])

  def getTrainingSample(self,batch_size, input_seq_length, output_seq_length):
    data = GenerateData.data
    input_batches = []
    output_batches = []
    n_starting_indexes = len(data) / (input_seq_length + output_seq_length)

    for i in range(batch_size):
      starting_index = np.random.randint(0,n_starting_indexes)
      starting_index_offset = starting_index * (input_seq_length + output_seq_length)

      an_input_batch_y = data[starting_index_offset: starting_index_offset + input_seq_length]
      input_batches.append(an_input_batch_y["y"])

      an_output_batch_y = data[starting_index_offset + input_seq_length:starting_index_offset + input_seq_length + output_seq_length]
      output_batches.append(an_output_batch_y["y"])
    return np.array(input_batches), np.array(output_batches)


  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))

#test:
#gd = GenerateData('univariate-data.csv')

#input_batches,output_batches  = gd.getTrainingSample(5,15,20)

#print input_batches, output_batches









 