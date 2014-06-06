from pickle import load
import os,sys,numpy as np,scipy as sp ,csv

#Deal with missing examples

def label_missing_numbers(np_array_obj):
	m,n = np_array_obj.shape
	new_array_obj=np.zeros_like(np_array_obj)
	for j in range(1,n):
		new_array_obj[:,j]=np.array([None if -999.1 <= np_array_obj[i,j] <= -998.9 \
			else np_array_obj[i,j] for i in range(m) ]).transpose()
	new_array_obj[:,0]=np_array_obj[:,0]
	return new_array_obj

def replace_missing_numbers(np_array_obj):
	m,n=np_array_obj.shape
	new_array=np.zeros_like(np_array_obj)
	for j in range(1,n):
		low=np.nanmin(np_array_obj[:,j])
		high=np.nanmax(np_array_obj[:,j])
		new_array[:,j]=np.array([np.random.uniform(low=low,high=high) if np.isnan(np_array_obj[i,j]) \
			else np_array_obj[i,j] for i in range(m)]).transpose()
	new_array[:,0]=np_array_obj[:,0]
	return new_array
#x_mean, x_sd=load(open("./x_mean_sd.pkl","rb"))

def find_mean_and_SD(np_array_obj,m,n):
	np_array_obj=np_array_obj[:,1:]
	colsum=np.zeros((1,n))
	mean=np.zeros((n,))
	sd=np.zeros((n,))

	try:
		colsum=sum(np_array_obj)

	except MemoryError:
		for i in range(m):
			colsum+=np_array_obj[k,:]

	mean=(1.0/m)*colsum

	for l in range(m):
		sd+=(np_array_obj[l,:]-mean)**2

	sd=(1.0/m)*sd

	return mean, sd

# Normalize number values of test data
def normalize_test_data(X ,m ,sd):

	m=m.reshape((1,len(m)))
	
	new_X=np.zeros_like(X)

	for i in range(X.shape[0]):
		new_X[i,1:]=(X[i,1:]-m)/sd
		new_X[i,0]=X[i,0]
	return new_X



# Score the result of a set of inputs

def score(tnet,inputs):
	Ps=np.zeros((2,1))
	outputs=tnet.activate(inputs)
	assert isinstance(outputs,np.ndarray), "Convert outputs to ndarray"
	Ps=outputs
	P=Ps[1]- Ps[0]
	return P

# Iterate over rows of test data, returning eventIDs, scores, and classes. 
def assemble_submission(test_x, trained_network,threshold):
	eventIDs=np.array([int(row[0]) for row in test_x])
	scores=np.array([score(trained_network, row) for row in test_x[:,1:]])
	classes=np.array(["s" if row>=(threshold-0.5) else "b" for row in scores])
	rank_order_ind=list(reversed(scores.argsort()))
	ranks=list(reversed(range(1,test_x.shape[0]+1)))
	submission=np.array([[eventIDs[val],ranks[i],classes[val]] for i,val in enumerate(rank_order_ind)])
	submission=np.append([["EventID","RankOrder","Class"]],submission,axis=0)

	return submission

#test_submission=assemble_submission(test_x_norm,trained_network)

def generate_diff_threshold_submissions():
	thresholds=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
	for index,threshold in enumerate(thresholds):
		np.savetxt("../../data/HIGGS_CHALLENGE/submission%d.csv"%(index+4), assemble_submission(test_x_norm,\
			trained_network,threshold), fmt='%s',delimiter=",")

def generate_pkl_files(dest):
	import re
	files=os.listdir(dest)
	res={}
	ind=1
	for item in enumerate(files):
		if re.search(r".[Pp][Kk][Ll]",item) is not None:
			res[ind]=item
			ind+=1

	indices=range(1,ind)

	return res,indices


def main():

	os.system("cls")
	#Load in Test Data
	all_test=list(csv.reader(open("../../data/HIGGS_CHALLENGE/test.csv"),delimiter=","))

	test_x=np.array([map(float,row) for row in all_test[1:]]) # [ EventID, {{X_var}} ]
	print "Loaded test data!"
	
	test_x=replace_missing_numbers(label_missing_numbers(test_x))
	print "Missing numbers replaced in test data!"
	
	x_mean,x_sd=find_mean_and_SD(test_x,test_x.shape[0],test_x.shape[1]-1)
	test_x_norm=normalize_test_data(test_x,x_mean,x_sd)
	print  "Data Normalized!"

	# Load in Trained Neural Network of old data

	pickle_files, index=generate_pkl_files("./")
	
	prompt_str="Choose from the following files:\n"
	for ind in index:
		prompt_str+="(%s) %s\n"%(ind,pickle_files[ind])
	prompt_str+="Input: "

	choice=pickle_files[int(raw_input(prompt_str))]
	trained_network=load(open(choice,"rb"))
	
	print "Network loaded!"
	submit=assemble_submission(test_x_norm,trained_network,0.5)
	
	print "Submission file assembled!"
	savename= raw_input("Savename?:")
	if savename[-4:].upper()==".CSV":
		savename=savename[:-4]

	np.savetxt("../../data/HIGGS_CHALLENGE/%s.csv"%(savename),submit,fmt="%s",delimiter=",")

if __name__=="__main__":
	main()




