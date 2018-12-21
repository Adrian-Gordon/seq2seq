
#output_multivariate_csv.py
#outputs a csv representation of some sample sequences in the form <sample no>,<observation no>, <x value>, <y1-value>, <y2-value>, <y3-value>, <y4-value>, <y5-value>
#usage: python output-univariate-csv.py <number of observations in a sample> <number of samples to generate> > <outfilename.csv>


import numpy as np
import sys

#from matplotlib import pyplot as plt


samples_per_sequence = int(sys.argv[1]) #length of the sample sequence
n_samples = int(sys.argv[2])            #number of samples to generate

class GenerateData:
  n_observations = samples_per_sequence * n_samples
  noise_factor = 0.5
  def __init__(self):
    self.x = np.linspace(0, GenerateData.n_observations, GenerateData.n_observations * 3)

    #self.train_data_x = self.x[:85]

    #target output values
    self.y1 = 2 * np.sin(self.x) + np.random.randn(GenerateData.n_observations* 3)* GenerateData.noise_factor
    self.y2 = 2 * np.cos(self.x) + np.random.randn(GenerateData.n_observations * 3)* GenerateData.noise_factor

    #input values

    self.y3 = (1.6*self.y1**4 - 2*self.y2 - 10) + np.random.randn(GenerateData.n_observations * 3)* GenerateData.noise_factor
    self.y4 = (1.2*self.y2**2 * self.y1 + 2*self.y2*3) - self.y1*6 + np.random.randn(GenerateData.n_observations * 3)* GenerateData.noise_factor
    self.y5 = (2*self.y1**3 + 2*self.y2**3 - self.y1*self.y2) + np.random.randn(GenerateData.n_observations * 3)* GenerateData.noise_factor


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
    data_y1 = self.y1[data_start_index:data_end_index]
    data_y2 = self.y2[data_start_index:data_end_index]
    data_y3 = self.y3[data_start_index:data_end_index]
    data_y4 = self.y4[data_start_index:data_end_index]
    data_y5 = self.y5[data_start_index:data_end_index]
    return data_x, data_y1,data_y2,data_y3,data_y4,data_y5


  def reshape(self, input_array, sequence_length, input_dimension):
    reshaped = [None]* sequence_length 

    for t in range(sequence_length):
      x = input_array[:,t].reshape(-1,input_dimension)
      reshaped[t]=x
    return(np.array(reshaped))

def main():
  gd = GenerateData()

  for i in range(n_samples):
   dx,dy1,dy2,dy3,dy4,dy5 = gd.getTrainingSample()
   for j in range(samples_per_sequence):
    print ('{0},{1},{2},{3},{4},{5},{6}'.format(((i * samples_per_sequence) + j + 1),dx[j],dy1[j],dy2[j],dy3[j],dy4[j],dy5[j]))
   #print dx, dy

 # l1, = plt.plot(gd.y1, 'r-',label = 'y1')
 # l2, = plt.plot(gd.y2, 'c-',label = 'y2')
 # l3, = plt.plot(gd.y5, 'c-',label = 'y5')
#  l2, = plt.plot(range(85, 105), gd.y[85:], 'yo', label = 'Test truth')
 # l3, = plt.plot(range(85,105), np.array(test_out).reshape(-1), 'r-', label = 'Test Predictions')
 # plt.legend(handles=[l3], loc = 'lower left')

 # plt.show()
main()




 