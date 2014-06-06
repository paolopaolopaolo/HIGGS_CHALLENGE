import random,string,math,csv,sys,os,time
import numpy as np
from pybrain.structure import FeedForwardNetwork, SigmoidLayer, LinearLayer, FullConnection, BiasUnit
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet, ImportanceDataSet
from pybrain.utilities import percentError
import matplotlib.pyplot as plt



# Thanks to the forums, we know that missing values are encoded by -999.0
def label_missing_numbers(np_array_obj):
	m,n = np_array_obj.shape
	new_array_obj=np.zeros_like(np_array_obj)
	for j in range(n):
		new_array_obj[:,j]=np.array([None if -999.1 <= np_array_obj[i,j] <= -998.9 \
			else np_array_obj[i,j] for i in range(m) ]).transpose()
	return new_array_obj

def replace_missing_numbers(np_array_obj):
	m,n=np_array_obj.shape
	new_array=np.zeros_like(np_array_obj)
	for j in range(n):
		low=np.nanmin(np_array_obj[:,j])
		high=np.nanmax(np_array_obj[:,j])
		new_array[:,j]=np.array([np.random.uniform(low=low,high=high)\
		 if np.isnan(np_array_obj[i,j]) else np_array_obj[i,j] for i in range(m)]).transpose()

	return new_array

def find_mean_and_SD(np_array_obj):
	m,n=np_array_obj.shape
	colsum=np.zeros((n,))
	mean=np.zeros((n,))
	sd=np.zeros((n,))
	working_array=np.zeros_like(np_array_obj)

	# Take all missing values and replace with 
	# mean of the appropriate column.
	for i in range(m):
		working_array[i]=np.array([0. if np.isnan(np_array_obj[i,j]) else np_array_obj[i,j] \
			for j in range(n)])
		colsum+=working_array[i]

	mean=(1.0/m)*colsum

	for l in range(m):
		sd+=(working_array[l]-mean)**2

	sd=(1.0/m)*sd

	return mean, sd

def meanNorm_featScalize(np_array_obj,mean,sd):
	new_array_obj=np.zeros((m,n))

	for i in range(m):

		new_array_obj[i]=(np_array_obj[i,:]-mean)/sd 

	return new_array_obj

#xs_mean,xs_sd=find_mean_and_SD(xs)

#xs=meanNorm_featScalize(xs,xs_mean,xs_sd,numPoints,numFeatures)


#print "Initial Mean:\n%s\nInitial SD:\n%s" % (str(xs_mean),str(xs_sd))

#new_mean,new_sd=find_mean_and_SD(xs)
#print "New Mean:\n%s\nNew SD:\n%s" % (str(new_mean),str(new_sd))


# COLUMN_PLOT(data, labels, col_num)
# Plot y='s' instances in red and y='b' instances in blue for identified column

def column_plot(data,labels,col_num,xmean,xsd):
	# clear plot
	plt.clf()
	
	# plot y='s' instances
	
	plt.subplot(2,1,1)
	#col_range=[xmean[col_num]-(3.*xsd[col_num]), xmean[col_num]+(3.*xsd[col_num])]

	plt.title("Columnwise Frequency Histogram for Higgs Data\n\t\tColumn:%d" %(col_num))
	plt.ylabel("s-freq")
	plt.xlabel("value")
	y_s_temp=[data[i,col_num] if labels[i]==1 else None for i in range(data.shape[0])]
	y_s_ins=[]
	for j in y_s_temp:
		if j is not None:
			y_s_ins.append(j)

	plt.hist(y_s_ins, bins=100, color="r",align="mid")
	
	# plot y='b' instances

	plt.subplot(2,1,2)
	plt.ylabel("b-freq")
	plt.xlabel("value")
	y_b_temp=[data[i,col_num] if labels[i]==0 else None for i in range(data.shape[0])]
	y_b_ins=[]
	for j in y_b_temp:
		if j is not None:
			y_b_ins.append(j)

	plt.hist(y_b_ins, bins=100, color="b",align="mid")
	plt.show()



def review_cols(x,y,xmean,xsd):
	#x_return=[]
	#x_reject=[]
	#retain=None
	#choice_dict={"Y":x_return.append,"N":x_reject.append}
	for k in range(x.shape[1]):
		os.system("cls")
		column_plot(x,y,k,xmean,xsd)
		#while retain not in choice_dict:
		#	retain=raw_input("Retain column %d (Y/N)?\nInput:"%(k+1)).upper()

		#choice_dict[retain](x[:,k])
		#retain=None

	#return np.array(x_return), np.array(x_reject)
	
#review_cols(xs,ys,xs_mean,xs_sd)		
#xs_normalized,xs_normalized_rejects=review_cols(xs_normalized,ys,new_mean,new_sd)

def main():
	os.system("cls")

	# Read in data
	all = list(csv.reader(open("../../data/HIGGS_CHALLENGE/training.csv","rb"), delimiter=','))

	# Slice off Header, EventID, Labels and Weights

	##DEBUGGING STEP HERE 
	xs = np.array([map(float, row[1:-2]) for row in all[1:]])

	# Helpful Variables
	numPoints, numFeatures = xs.shape

	# Perturb certain cells to avoid ties
	xs = np.add(xs, np.random.normal(0.0, 0.0001, xs.shape))

	# Set S-selector and B-selector 
	sSelector = np.array([row[-1] == 's' for row in all[1:]])
	bSelector = np.array([row[-1] == 'b' for row in all[1:]])

	# Set y=0 for B examples and y=1 for S examples

	ys=np.array([1 if row[-1]=='s' else 0 for row in all[1:]])

	# Set weights and weight sums

	weights = np.array([float(row[-2]) for row in all[1:]])
	sumWeights = np.sum(weights)
	sumSWeights = np.sum(weights[sSelector])
	sumBWeights = np.sum(weights[bSelector])

	print "Data Loaded!"
	xs=label_missing_numbers(xs)
	print "Missing values replaced with 'nan'"

	
	xs=replace_missing_numbers(xs)
	print "'nan' replaced with random value of uniform distribution"

	return xs, ys, sumWeights, sumSWeights, sumBWeights


if __name__="__main__":
	xs,ys,sumWeights,sumSWeights,sumBWeights=main()

