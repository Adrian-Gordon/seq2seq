
#output-univariate-csv.py
#outputs a csv representation of some sample sequences in the form <sample no>,<observation no>, <x value>, <y-value>
#python usage output-univariate-csv.py <number of observations in a sample> <number of samples to generate> > <outfilename.csv>


import numpy as np
import sys


samples_per_sequence = int(sys.argv[1]) #length of the sample sequence
n_samples = int(sys.argv[2])            #number of samples to generate

class GenerateData:
  n_observations = samples_per_sequence * n_samples
  noise_factor = 0.5
  def __init__(self):
    self.x = np.linspace(0, GenerateData.n_observations, GenerateData.n_observations * 3)

    #self.train_data_x = self.x[:85]

    self.y = 2 * np.sin(self.x) + np.random.randn(GenerateData.n_observations * 3)* GenerateData.noise_factor


    #self.train_y = self.y.copy()

    #self.noise_factor = 0.5

    #self.train_y += np.random.randn(105)* self.noise_factor



  def true_signal(self,x):
    y = 2 * np.sin(x)
    return(y)

  def noise_func(self,x, noise_factor = 0.5):
    return np.random.randn(len(x)) * self.noise_factor

  def generate_y_values(self,x):
    return self.true_signal(x) + self.noise_func(x)


  def getTrainingSample(self):
    n_startingIndexes = len(self.x) / samples_per_sequence
    sample_index = np.random.randint(0,n_startingIndexes)
    data_start_index = sample_index * samples_per_sequence
    data_end_index = data_start_index + samples_per_sequence
    data_x = self.x[data_start_index:data_end_index]
    data_y = self.y[data_start_index:data_end_index]
    return data_x, data_y


  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))

def main():
  gd = GenerateData()

  for i in range(n_samples):
   dx,dy = gd.getTrainingSample()
   for j in range(samples_per_sequence):
    print ('{0},{1},{2}'.format(((i * samples_per_sequence) + j + 1),dx[j],dy[j]))
   #print dx, dy


main()




 