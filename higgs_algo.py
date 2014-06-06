
# NNET returns Feedforward Neural Network Object trained with Backpropagation and partitioned datasets
def nnet(X,y,input_layer=None, hidden_layer_num=None, hidden_layers=None, output_layer=None,\
	class_labels=None, importance=None,w=weights,sum_w=sumWeights,\
	sum_s=sumSWeights,sum_b=sumBWeights, bias=False,learningrate=0.01,iters=None,lrdecay=1.0,\
	eval_LC=False,mean_norm=False,feature_scale=False,mdebug=None):

	##########################  -CREATE NEURAL NETWORK ARCHITECTURE   #################################

	#  Set Default Architecture
	m,n=X.shape
	parameters=[input_layer,hidden_layer_num,hidden_layers,output_layer,class_labels,importance]
	defaults=[[input_layer,X.shape[1]], [hidden_layer_num,2], [hidden_layers, [X.shape[1]+2, \
		X.shape[1]+2]], [output_layer,2],[class_labels,[0,1]],[importance,np.array([(sum_s/sum_w),\
			(sum_b/sum_w)])]]

	for item in defaults:
		if item[0] is None:
			item[0]=item[1]

	for i in range(len(parameters)):
		parameters[i] = defaults[i][0]

	input_layer=parameters[0]
	hidden_layer_num=parameters[1]
	hidden_layers=parameters[2]
	output_layer=parameters[3]
	class_labels=parameters[4]
	importance=parameters[5]

	print "Architecture parameters set!"

	#  Input and Output layers

	fnn=FeedForwardNetwork()
	fnn.addInputModule(LinearLayer(input_layer,name="Input"))
	fnn.addModule(BiasUnit(name="Bias"))
	fnn.addOutputModule(SigmoidLayer(output_layer,name="Output"))

	print "I/O Layers set and added to architecture!"

	#  Hidden Layers

	try:
		for i in range(0,hidden_layer_num):
			fnn.addModule(SigmoidLayer(hidden_layers[i], name="Hid_"+str(i+1)))
			
	except IndexError:
		print "Indexing Error: hidden_layer_num does not match number of hidden_layers"

	print "Hidden layers set and added to architecture!"

	#  Connect everything together

	connections=[]
	in_hid_layers=FullConnection(fnn["Input"],fnn["Hid_1"],name="in_to_hid")
	#in_bias_unit=FullConnection(fnn["Input"],fnn["Bias"],name="in_to_bias")
	connections.append(in_hid_layers)
	#connections.append(in_bias_unit)

	try:
		indict={}
		for j in range(0,hidden_layer_num-1):
			indict[j]=FullConnection(fnn["Hid_%d"%(j+1)],fnn["Hid_%d"%(j+2)],\
				name="hid%d_to_hid%d"%(j+1,j+2))
			indict[j+100]=FullConnection(fnn["Bias"],fnn["Hid_%d"%(j+1)],name="bias_hid%d"%(j+1))
			connections.append(indict[j])
			connections.append(indict[j+100])

	except IndexError:
		print "Indexing Error: hidden_layer_num does not match number of hidden_layers"

	hid_layers_out=FullConnection(fnn["Hid_%d"%(hidden_layer_num)],fnn["Output"],name="hid_to_out")
	bias_last_hid_layer=FullConnection(fnn["Bias"],fnn["Hid_%d"%(hidden_layer_num)],name="bias_hid%d"%(hidden_layer_num))
	connections.append(hid_layers_out)
	connections.append(bias_last_hid_layer)

	for connection in connections:
		fnn.addConnection(connection)
		print str(connection)+" added!"

	# Consolidate Structure
	fnn.sortModules()
	fnn.randomize()
	print "Modules sorted! Neural Network is ready to use!"

	
	##########################  -FEED X,y into DATASET OBJECT         #################################

	#initialize DS object

	DS=ClassificationDataSet(inp=input_layer, target=1, nb_classes=output_layer, class_labels=class_labels)

	# Add Field for Importance and Weights

	w=w[:m]
	DS.addField("importance",dim=output_layer)
	DS.addField("weights",dim=1)
	DS.linkFields(["input","target","importance","weights"])
	

	# Feature Scaling and Mean Normalization and Bias Unit Appenthesis
	def find_mean_and_SD(np_array_obj,m,n):
		colsum=np.zeros((1,n))
		mean=np.zeros((1,n))
		sd=np.zeros((1,n))

		try:
			colsum=sum(np_array_obj)

		except MemoryError:
			for i in range(m):
				colsum+=np_array_obj[k,:]

		mean=(1.0/m)*colsum

		for l in range(m):
			sd+=(np_array_obj[l,:]-mean)**2

		sd=(1.0/m)*sd

		return mean,sd

	def meanNorm_featScalize(np_array_obj,mean,sd,mean_norm,feature_scale):
		m,n=np_array_obj.shape
		new_array_obj=np.zeros((m,n))
		temp_row=np.zeros((1,n))
		for i in range(m):
			temp_row=np_array_obj[i]
			if mean_norm is True:
				temp_row-=mean
			if feature_scale is True:
				temp_row=temp_row/sd
			new_array_obj[i]=temp_row

		return new_array_obj
		
	mu , sigma = find_mean_and_SD(X,m,n)	
	X = meanNorm_featScalize(X,mu,sigma,mean_norm,feature_scale)
	
	print "Mean and SD of each column in dataset found!"
	print "Dataset run through mean normalization (%s) and feature scaling (%s) function!"%(mean_norm,feature_scale)


	# Load X, y and w into dataset

	for r in range(m):
		print "\b"*500+"%d/%d rows of data loaded into dataset!"%(r+1,m),
		DS.appendLinked( X[r], y[r] , importance , w[r])
	print "\n"

	
	# Segment training and cv datasets

	DSCV,DSTrain = DS.splitWithProportion(0.25)

	# Scale weights evenly among DSTrain and DSCV
	DSCV_scaled_weights=DSCV["weights"]*(sum_w/sum(DSCV["weights"]))
	DSTrain_scaled_weights=DSTrain["weights"]*(sum_w/sum(DSTrain["weights"]))

	DSCV.setField("weights_scaled",DSCV_scaled_weights)
	DSTrain.setField("weights_scaled",DSTrain_scaled_weights)

	# Convert y into 2-element vectors. y=0 => y=[1 0] and y=1 => y=[0 1] 
	DSCV._convertToOneOfMany(bounds=[0,1])
	DSTrain._convertToOneOfMany(bounds=[0,1])


	##########################  -TRAIN NNet with BACKPROP TRAINER     #################################

	#######DEBUG: Set learning rate here##############################
	trainer=BackpropTrainer(fnn, dataset=DSTrain,learningrate=learningrate,verbose=True,\
		lrdecay=lrdecay,eval_LC=eval_LC)
	print "Backprop Trainer Initialized!" 

	def train_nnet(backproptrainer_obj,iterations=iters):
		if iterations is None:
			try:
				iterations=int(raw_input("How many iterations will be performed?\nInput: "))

			except (ValueError,TypeError):
				train_nnet(backproptrainer_obj)

		for i in range(iterations):
			print "Iteration %d of %d:" % (i+1,iterations)
			backproptrainer_obj.train()

			#train_error=percentError(trainer.testOnClassData(), datasetTrain['class'])
			#CV_error=percentError(trainer.testOnClassData(), datasetCV['class'])

			#print "Training Error: %5.2f%%\t" % train_error,
			#print "CV Error: %5.2f%%" % CV_error

	train_nnet(trainer)

	##########################  -RETURN NNet Object and DataSets  #################################

	return trainer, DSTrain, DSCV, DS, mu, sigma

### Score will evaluate the probability point diff of s over b for any
### given input
def score(tnet,inputs):
	Ps=np.zeros((2,1))
	outputs=tnet.activate(inputs)
	assert isinstance(outputs,np.ndarray), "Convert outputs to ndarray"
	Ps=outputs
	P=Ps[1] - Ps[0]
	return P

### FIND_S_AND_B  will determine s and b values of cross validation data
def find_s_and_b(trained_network,dataset=None,threshold=0.5,stats=False,verbose=False):
	# Finds s and b of cross validation data

	#trainyhat,traintarg=trained_network.testOnClassData(return_targets=True)
	if dataset is not None:
		scores = [score(trained_network.module, row) for row in dataset["input"]]
		yhat = [1. if row >= (threshold-0.5) else 0. for row in scores]

	else:
		raise Exception, "Fill in dataset parameter!"

	s=sum([dataset["weights_scaled"][i] if yhat[i]==dataset["class"][i] and yhat[i]==1 else 0 for i in range(len(yhat))])
	b=sum([dataset["weights_scaled"][i] if yhat[i]!=dataset["class"][i] and yhat[i]==1 else 0 for i in range(len(yhat))])

	#strain=sum([traindata["weights"][i] if trainyhat[i]==traindata["class"][i] and trainyhat[i]==1 \
	#	else 0 for i in range(len(trainyhat))])
	#btrain=sum([traindata["weights"][i] if trainyhat[i]!=traindata["class"][i] and trainyhat[i]==1 \
	#	else 0 for i in range(len(trainyhat))])

	if stats is True:
		#trainerror=percentError(trainyhat,traindata["class"])
		testerror=percentError(yhat, dataset["class"])

		#print "TrainError:%5.2f%%" % trainerror
		print "TestError:%5.2f%%" % testerror

	if verbose is True:
		#print "s of train set/total weights:%6.2f/%6.2f" % (strain,sum([traindata["weights"][i] \
		#	if traindata["class"][i]==1 else 0 for i in range(len(trainyhat))]))
		#print "b of train set/total weights:%6.2f/%6.2f" % (btrain,sum([traindata["weights"][i] \
		#	if traindata["class"][i]==0 else 0 for i in range(len(trainyhat))]))

		print "s of dataset/total s weights:%6.2f/%6.2f" % (s,sum([dataset["weights_scaled"][i] \
			if dataset["class"][i]==1 else 0 for i in range(len(yhat))]))
		print "b of dataset/total b weights:%6.2f/%6.2f" % (b,sum([dataset["weights_scaled"][i] \
			if dataset["class"][i]==0 else 0 for i in range(len(yhat))]))

	return float(s),float(b)

### AMS evaluation
def AMS(s,b):
	s=float(s)
	b=float(b)
	res=(2.*(((s+b+10.)*(np.log(1.+(s/(b+10.)))))-s))**(0.5)
	return res

### FIND_BEST_IMPORTANCE will determine best importance values to plug into learning algorithm
### that will be used to evaluate the test data
def find_best_Importance(iters=None):
	### Call Neural Network Function: Obtain Trained Network and Test Dataset
	### Call with different importance weights 
	a=np.array(range(10,110,10))/100.
	b=np.ones_like(a)
	importance_factors=zip(a,b)
	ss_and_bs=[]
	max_importance=None
	max_AMS=0.

	for importance_factor in importance_factors:
		os.system("cls")
		print "Training and Testing with Importances: Imp_b~%3.2f , Imp_s~%3.2f:" % \
		(importance_factor)

		#################### DEBUGGING STEP: make m<250000 ######################################
		tnet, tdata, cdata, adata, xmean, xsd = nnet(xs,ys,hidden_layer_num=10,\
			hidden_layers=[50,50,50,50,50,50,50,50,50,50],output_layer=2, importance=np.array(importance_factor),\
			 class_labels=["b","s"],iters=iters,feature_scale=True,mean_norm=True,eval_LC=True)

		# FIXME: Find a way to propagate mean and sd from columns up 
		# so that they can get used by the testing function in
		# preprocessing the test data

		stemp, btemp = find_s_and_b(tnet, dataset=cdata, stats=True, verbose=True)
		ss_and_bs.append((importance_factor,stemp,btemp,tnet))
		del tnet
		del tdata
		del cdata
		del adata
		del xmean
		del xsd

	score_table=[]

	for item in ss_and_bs:
		ams=AMS(item[1],item[2])
		if ams > max_AMS:
			max_AMS=ams
			max_importance=item[0]
			max_tnet=item[3]
		score_table.append((ams,item[0]))

	print "max_AMS: %6.4f" % max_AMS
	print "importance at max_AMS: %s" % str(max_importance) 
	return max_AMS, max_importance, max_tnet, score_table


def find_best_threshold(*args):
	thresholds=list(args)
	score_table=[]
	max_importance=(0.5,1.0)
	max_AMS=-1.0
	max_threshold=None

	trainnet, traindata, cvdata, alldata, x_mean, x_sd = nnet(xs,ys,hidden_layer_num=10,\
		hidden_layers=[50,50,50,50,50,50,50,50,50,50],output_layer=2, class_labels=["b","s"],iters=1,\
		eval_LC=True,importance=np.array(max_importance),learningrate=0.001,mean_norm=True,\
		feature_scale=True)	

	for threshold in thresholds:
		print "Testing Threshold:%6.2f"%(threshold),
		scv,bcv=find_s_and_b(trainnet,dataset=cvdata,threshold=threshold)
		ams=AMS(scv,bcv)
		print  "\tAMS=%6.4f"%(ams)
		print "scv:%s, bcv:%s"%(scv,bcv)
		score_table.append((threshold,ams))
		if ams>max_AMS:
			max_AMS=ams
			max_threshold=threshold

	print "max_AMS: %6.4f"%(max_AMS)
	print "max_threshold: %6.2f"%(max_threshold)
	return max_AMS,max_threshold,trainnet.module,score_table

#max_AMS, max_threshold, trainnet, score_table=find_best_threshold(0.01,0.02,0.03,0.04,
#	0.05,0.06,0.07,0.08,0.09,0.10)

def save():
	from pickle import dump
	import re
	trainnet=trainnet.module
	ams_str=re.search( r'([\d]*).([\d]+)',str(ams))
	dump(trainnet,open("./trained_network_AMS_%s_%s.pkl"%(ams_str.group(1),ams_str.group(2)),"wb"))



def main():
	if sys.argv[1]=="run_with_defaults" or sys.argv[1]="":

		trainnet, traindata, cvdata, alldata, x_mean, x_sd = nnet(xs,ys,hidden_layer_num=10,\
		hidden_layers=[50,50,50,50,50,50,50,50,50,50],output_layer=2, class_labels=["b","s"],iters=5,\
		eval_LC=True,learningrate=0.001,mean_norm=True,feature_scale=True)	

		scv,bcv=find_s_and_b(trainnet,dataset=cvdata,stats=True,verbose=True)
		ams=AMS(scv,bcv)
		
		choice_dict={'Y':save,'N':sys.exit}
		savefile=raw_input("Save trained_network (Y/N)?\nInput: ").upper()

		try:
			choice_dict[savefile]()
		except (KeyError):
			save()


if __name__="__main__":
	main()



#print max_AMS
#plt.plot(trainnet.JHistory)
#plt.title("Cost Averaged every 1000 Examples")
#plt.ylabel("Cost")
#plt.xlabel("Examples (1000s)")
#plt.show()

