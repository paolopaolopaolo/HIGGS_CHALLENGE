import numpy as np, sys, scipy as sp, csv,random
from scipy import optimize

class Logistic_Regression_Model():
	""" 
	Logistic regression is a supervised learning 
	method that allows one to automate binary 
	classification. 

	The Logistic Regression Model takes numpy 
	arrays X (training data) and y (teaching data) 
	and prepares these arrays for logistic regression 
	training. 

	The train method can be interactive and provides 
	the choice of using either gradient descent or a
	host of advanced optimization techniques for 
	approaching the optimal parameters. 

	The test method only works with binary
	classification, but will provide a number square 
	that identifies the numbers of accurate guesses, 
	false positives and false negatives.

	"""

	m=None
	X=None
	y=None
	theta=None
	alpha=None
	lamb=None
	JHistory=[]
	res=None

	def __init__(self , X , y , mean_norm=None):

		"""Initialize internal arrays, preprocess 
		with mean normalization if mean_norm==True"""

		try:
			self.m=len(y)
			#self.X=X
			self.X=np.hstack((np.ones((self.m,1)),X))
			self.y=y.reshape((self.m,1))
			self.theta=np.zeros((len(self.X[0]),1))

		except TypeError:
			print "Load X and y to workspace before accessing model!"

		if mean_norm is None:
			appropriate_responses=["Y","N"]
			fn_y_n=raw_input("Would you like X's features to be mean normalized(Y/N)?\nInput:").upper()

			if fn_y_n in appropriate_responses:

				if fn_y_n=="Y":

					for i in range(1,len(self.X[0])):
						self.X[:,i]=(self.X[:,i]-(sum(self.X[:,i])/len(self.X[:,i])))/(np.max(self.X[:,i])-np.min(self.X[:,i]))
				else:
					pass

			else:
				print "Error! Try again!"
				self.__init__(X,y)

		elif mean_norm is True:
			for i in range(1,len(self.X[0])):
				self.X[:,i]=(self.X[:,i]-(sum(self.X[:,i])/len(self.X[:,i])))/(np.max(self.X[:,i])-np.min(self.X[:,i]))

		else:
			print "Warning: X not Mean Normalized.\n"



	def sigmoid(self,z):
		"""Internal sigmoidal function. Takes value z and 
		feeds it into logistic sigmoid function. The sigmoid 
		function asymptotes at y=0 and y=1 with its y-intercept 
		at 0.5."""

		return 1/(1+np.e**-z)

	def linear_hypothesis(self,X,theta):
		"""Internal linear hypothesis function. Takes X and 
		theta parameters and returns Theta^transpose*X """
		return (np.dot(theta.transpose(),X.transpose())).transpose()

	def logistic_hypothesis(self,X,theta):
		"""Internal logistic hypothesis function. Feeds the 
		results of the linear hypothesis function into the 
		logistic sigmoid function."""
		return self.sigmoid(self.linear_hypothesis(X,theta))

	def cost_function(self,theta,X,y,m):
		"""Internal cost function. Log loss function."""
		y1cost=y*np.log(self.logistic_hypothesis(X,theta))
		y0cost=(1-y)*np.log(1-self.logistic_hypothesis(X,theta))
		return float((-1/m)*sum(y1cost+y0cost))

		
	def gradient_descent(self,iterations=400,alpha=None):
		"""Internal gradient descent algorithm. Performs gradient descent
		to optimize the parameters for minimizing the cost function.
		"""

		try:
			self.alpha=float(raw_input("Choose learning rate: "))
		except (ValueError,TypeError):
			print "Try again!"
			self.gradient_descent()

		for iteration in range(0,iterations):
			#OCTAVE: theta = theta -((alpha/m)*((linear_hypothesis(X,theta)-y)'*X)');
			self.theta=self.theta-(self.alpha/self.m)*np.dot(((self.logistic_hypothesis(self.X,self.theta)-self.y)).transpose(),self.X).transpose()
			self.JHistory.append(self.cost_function(self.theta,self.X,self.y,self.m))

		
	def BFGS(self,cost,gradient):
		"""Internal call to BFGS algorithm, which optimizes the 
		parameters for minimizing the cost function."""

		self.theta=optimize.fmin_bfgs(cost, x0=self.theta, fprime=gradient,args=(self.X,self.y,self.m,self.lamb))

	def LBFGS(self,cost,gradient):
		"""Internal call to LBFGS algorithm, which optimizes the
		parameters for minimizing the cost function."""

		self.theta=optimize.fmin_l_bfgs_b(cost, x0=self.theta , fprime=gradient, args=(self.X,self.y,self.m,self.lamb))[0]

	def conjugate_gradient(self,cost,gradient):
		"""Internal call to Conjgate Gradient algorithm, which
		optimizes the parameters for minimizing the cost function."""

		self.theta=optimize.fmin_cg(cost, x0=self.theta, fprime=gradient, args=(self.X,self.y,self.m,self.lamb))
		#print cost(self.theta,self.X,self.y,self.m)
		

	def adv_opt(self, opt=None,lamb=0):
		"""Internal advanced optimization menu. If opt is None, 
		will allow interactive selection of advanced optimization 
		technique. Calculates both cost function and gradient to 
		be fed into the advanced optimization algorithms."""

		def sig(z):
			return 1/(1+np.e**-z)

		def lin_hyp(X,theta):
			return (np.dot(theta.transpose(),X.transpose())).transpose()

		def log_hyp(X,theta):
			return sig(lin_hyp(X,theta))

		options=["(0) Exit","(1) Conjugate Descent","(2) BFGS","(3) L-BFGS"]
		options_dict={0:sys.exit, 1:self.conjugate_gradient, 2:self.BFGS, 3:self.LBFGS}

		if opt is None:
			optiontext="The following advanced techniques are available:"
			for option in options:
				optiontext+="\n"+option
			optiontext+="\nInput: "
			try:
				choice=int(raw_input(optiontext))

			except (TypeError,ValueError,IndexError):
				print "Not a valid choice, try again!"
				self.train()

		##### COST AND GRADIENT FUNCTIONS#####
		def cost(theta,X,y,m,lamb):
			theta=theta.reshape((len(theta),))
			X=X.reshape((m,len(X[0])))
			y=y.reshape((m,))
			y1cost=y*np.log(log_hyp(X,theta))
			y0cost=(1.0-y)*np.log(1.0-log_hyp(X,theta))
			return ((-1.0/m)*(sum(y1cost+y0cost)+(lamb/(2*m))*sum(theta[1:]**2)))
			

		def gradient(theta,X,y,m,lamb):
			theta=theta.reshape((len(theta),))
			X=X.reshape((m,len(X[0])))
			y=y.reshape((m,))
			result=(1.0/m)*np.dot(X.transpose(), log_hyp(X,theta)-y)+(lamb/m)*np.hstack(([0],theta[1:]))
			return result
	
		##### RUN THE APPROPRIATE OPTIM FUNCTION ####
		if opt is None:
			if choice != 0:
				options_dict[choice](cost,gradient)
			else:
				options_dict[choice](choice)
		elif opt=="CG":
			options_dict[1](cost, gradient)
		elif opt=="BFGS":
			options_dict[2](cost,gradient)
		elif opt=="LBFGS":
			options_dict[3](cost,gradient)
		else:
			self.adv_opt(opt=None)


	def train(self, opt=None, alpha=None, lamb=0):
		"""The train method for the Logistic Regression 
		model object. Providing opt will remove need for further 
		interactive input. Learning rate (alpha) is only necessary 
		if opt=='GD' """

		self.lamb=lamb
		opts=["GD","CG","BFGS","LBFGS"]

		if opt is None:
			options=["(0) Exit","(1) Gradient Descent","(2) Advanced Optimization"]
			options_dict={0:sys.exit, 1:self.gradient_descent, 2:self.adv_opt}
			optiontext="The following optim techniques are available for the logistic regression model:"

			for option in options:
				optiontext+="\n"+option
			optiontext+="\nInput: "

			try:
				choice=int(raw_input(optiontext))

			except (TypeError,ValueError,IndexError):
				print "Not a valid choice, try again!"
				self.train()

			if choice != 0:
				options_dict[choice]()
			else:
				options_dict[choice](choice)
		
		elif opt in opts:
			if opt=="GD":
				self.gradient_descent(alpha=alpha)
			elif opt=="BFGS" or opt=="CG" or opt=="LBFGS":
				self.adv_opt(opt=opt)
			else:
				self.train(opt=None)
		else:
			self.train(opt=None)

	def test(self, test_set_filename):
		data=np.loadtxt(test_set_filename, delimiter=",")

		m=data.shape[0]
		tries=m
		
		try:
			ycol=int(raw_input("What is the y-col for the test set?"))
		
		except TypeError:
			print "Try again!"
			self.test(test_set_filename)

		y=data[:,ycol]
		X=sp.delete(data,ycol,1)
		X=np.hstack((np.ones((self.m,1)), X))

		hx0y0=0
		hx1y0=0
		hx0y1=0
		hx1y1=0
		other=0

		for i in range(0,tries):
			if (self.logistic_hypothesis(X[i,:],self.theta) >=0.5 and y[i] ==1):
				hx1y1+=1
			elif (self.logistic_hypothesis(X[i,:],self.theta) <0.5 and y[i] ==0):
				hx0y0+=1
			elif (self.logistic_hypothesis(X[i,:],self.theta) >= 0.5 and y[i] ==0):
				hx1y0+=1
			elif (self.logistic_hypothesis(X[i,:],self.theta) < 0.5 and y[i] ==1):
				hx0y1+=1
			else:
				other+=1


		print "Accuracy on test data:\n"
		print "hx\t0\t1"
		print "y 0\t"+str(hx0y0)+"\t"+str(hx1y0)
		print "  1\t"+str(hx0y1)+"\t"+str(hx1y1)
		print "Other: "+str(other)
		print "Accuracy: %02.2f" % float((float(hx1y1+hx0y0)/float(tries))*100.)


class Neural_Network_Model():
	"""

	Neural network is an advanced supervised machine learning
	classification method that utilizes a "hidden layer" to create
	complex calculation models, enabling the generation of non-linear 
	hypotheses. 

	This model will use the sigmoidal logistic function as the 
	activation function and comes with a choice of 1,2 or 3 
	hidden layers of varying node number. The backpropagation
	algorithm is used to train the neural network

	"""

	## Architecture ##
	data_file=""
	m=0
	lamb=0
	y_col=0
	input_nodes=0
	hidden_layer_num=0
	hidden_layer_nodes=[]
	output_nodes=0

	## Parameters ##
	init_nnparams=None
	nnparams=None
	Theta1_par=(0,0)
	Theta2_par=(0,0)
	Theta3_par=(0,0)
	ThetaEnd_par=None

	def mult_tuple(self,tuple):
		res=1
		for i in tuple:
			res*=i
		return res

	def set_architecture(self):
		try:
			self.input_nodes=int(raw_input("How many features are in X?: "))
			self.hidden_layer_num=int(raw_input("How many hidden layers will you use (1-3)?: "))

			if self.hidden_layer_num < 1 or self.hidden_layer_num>3:
				self.hidden_layer_num=1
				print "Warning: Invalid value. 1 hidden layer selected."
			
			self.hidden_layer_nodes=[]
			for i in range(0,self.hidden_layer_num):
				add_nodes=int(raw_input("How many nodes will be added to hidden layer "+str(i+1)+"?: "))

				self.hidden_layer_nodes.append(self.input_nodes+add_nodes)

			self.output_nodes=int(raw_input("How many distinct classes are in the dataset?: "))

		except (ValueError,TypeError):
			print "Invalid values! Try again!"
			self.set_architecture()

	def randomize_nnparams(self):

		eps=((6.0)**(0.5))/((float(self.input_nodes)+float(self.output_nodes))**(0.5))

		self.Theta1_par=(self.hidden_layer_nodes[0], self.input_nodes+1)
		self.ThetaEnd_par=(self.output_nodes, self.hidden_layer_nodes[-1]+1)

		if self.hidden_layer_num>1:

			self.Theta2_par=(self.hidden_layer_nodes[1],self.hidden_layer_nodes[0]+1)

			if self.hidden_layer_num>2:

				self.Theta3_par=(self.hidden_layer_nodes[2],self.hidden_layer_nodes[1]+1)
				
		self.init_nnparams=np.random.rand(self.mult_tuple(self.Theta1_par)+self.mult_tuple(self.ThetaEnd_par)+self.mult_tuple(self.Theta2_par)+self.mult_tuple(self.Theta3_par),)*(2*eps)-eps
		self.nnparams=np.zeros_like(self.init_nnparams)


	def __init__(self, filename, y_col, m=None, input_nodes=0, hidden_layer_num=0, hidden_layer_nodes=[], output_nodes=0):
		"""filenames= the string path of the CSV data to be fed into the model
		y_col= integer value of the column of the training instance (0 indexed)
		hidden_layer_num= number of hidden layers (can be 1-3)
		hidden_layer_nodes= list corresponding to the number of nodes in 
		                    each hidden layer
		output_nodes= number of output nodes"""

		#checks number of rows in file
		if m is None:
			infile=open(filename,'rb')
			for line in infile:
				self.m+=1
			infile.close()
		else:
			self.m=m

		if self.data_file!="":
			self.data_file=""

		
		self.data_file+=filename
		self.y_col= y_col
		self.input_nodes=input_nodes
		self.hidden_layer_num=hidden_layer_num
		self.hidden_layer_nodes=hidden_layer_nodes

		if self.hidden_layer_num != len(hidden_layer_nodes):
			raise IndexError
		self.output_nodes=output_nodes
		
		if input_nodes==0 or hidden_layer_num==0 or len(hidden_layer_nodes)==0 or output_nodes==0:
			self.set_architecture()

		self.randomize_nnparams()
			
		
	def review_architecture(self):
		print "~~Architecture~~"
		print "Number of Examples: "+str(self.m)
		print "Input Nodes: "+str(self.input_nodes)
		hid_layer_count=1
		for num in self.hidden_layer_nodes:
			print "Hidden Layer "+str(hid_layer_count)+": "+str(num)
			hid_layer_count+=1
		print "Output Nodes: "+str(self.output_nodes)

	def __call__(self, filename):
		pass

	def train(self, opt=None,lamb=0):

		self.lamb=lamb

		def sigmoidGradient(z):
			return (1.0/(1.0+np.e**-z))*(1.0-(1.0/1.0+np.e**-z))

		def sigmoid(z):
			return 1.0/(1.0+np.e**-z)

		other_args=(self.data_file,self.y_col,self.m,self.lamb,self.Theta1_par,self.Theta2_par,self.Theta3_par,self.ThetaEnd_par,self.input_nodes,self.hidden_layer_num,self.hidden_layer_nodes,self.output_nodes)

		def cost(nnparams,data_file,y_col,m,lamb,Theta1_par,Theta2_par,Theta3_par,ThetaEnd_par,input_nodes,hidden_layer_num, hidden_layer_nodes, output_nodes):

			def mult_tuple(tupl):
				res=1
				for i in tupl:
					res*=i
				return res

			with open(data_file,"rb") as f: # self.data_file
				row_read=csv.reader(f,delimiter=",")

				ymap=np.eye(output_nodes,dtype=float) # self.output_nodes
				Theta1=nnparams[0:(mult_tuple(Theta1_par))].reshape(Theta1_par) # self.nnparams, self.Theta1_par
				Theta2=0
				Theta3=0
				ThetaEnd=0

				#Populate parameters with randomized values (randomized in randomize_nnparams method).

				if hidden_layer_num==1: #self.hidden_layer_num
					ThetaEnd=nnparams[(mult_tuple(Theta1_par)):(mult_tuple(Theta1_par)+mult_tuple(ThetaEnd_par))].reshape(ThetaEnd_par)
					
				elif hidden_layer_num==2:
					Theta2=nnparams[(mult_tuple(Theta1_par)):(mult_tuple(Theta1_par)+mult_tuple(Theta2_par))].reshape(Theta2_par)
					ThetaEnd=nnparams[(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)):(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)+mult_tuple(ThetaEnd_par)+1)].reshape(ThetaEnd_par)

				else:
					Theta2=nnparams[(mult_tuple(Theta1_par)):(mult_tuple(Theta1_par)+mult_tuple(Theta2_par))].reshape(Theta2_par)
					Theta3=nnparams[(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)):(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)+mult_tuple(Theta3_par))].reshape(Theta3_par)
					ThetaEnd=nnparams[(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)+mult_tuple(Theta3_par)):(mult_tuple(Theta1_par)+mult_tuple(Theta2_par)+mult_tuple(Theta3_par)+mult_tuple(ThetaEnd_par)+1)].reshape(ThetaEnd_par)
					
				J=0
				grad=np.zeros_like(nnparams)

				Theta1_grad=np.zeros_like(Theta1)
				ThetaEnd_grad=np.zeros_like(ThetaEnd)
				Theta2_grad=0
				Theta3_grad=0

				if hidden_layer_num > 1:
					Theta2_grad=np.zeros_like(Theta2)

					if hidden_layer_num == 3:
						Theta3_grad=np.zeros_like(Theta3)

				for i in range(0,m):
					#Load in Row
					ROW=row_read.next()
					data=np.array(ROW, dtype=float)
					y=ymap[:,int(data[y_col])].reshape((output_nodes,1)) # output_nodes x 1
					X=np.hstack((1.0,sp.delete(data,y_col))) # 1 x (n+1)
			
					#Forward Propagation
					z2=np.dot(Theta1,X.reshape((len(X),1)))  #(hidden_layer_node[0] x n+1) X (n+1 x 1) = hidden_layer_node[0] x 1		
					a2=np.vstack((1.0, sigmoidGradient(z2))) # (hidden_layer_node [0] + 1 )x 1
					aEnd=np.zeros((output_nodes,1))
					if hidden_layer_num==1:
						z3=np.dot(ThetaEnd,a2) # (output_node x (hidden_layer_node [0] + 1 )) x ((hidden_layer_node [0] + 1) x 1)
						a3=sigmoid(z3)
						aEnd=a3

					elif hidden_layer_num==2:
						z3=np.dot(Theta2,a2)
						a3=np.vstack((1.0,sigmoid(z3)))
						z4=np.dot(ThetaEnd,a3)
						a4=sigmoid(z4)
						aEnd=a4

					else:
						z3=np.dot(Theta2,a2)
						a3=np.vstack((1.0,sigmoid(z3)))
						z4=np.dot(Theta3,a3)
						a4=np.vstack((1.0,sigmoid(z4)))
						z5=np.dot(ThetaEnd,a4)
						a5=sigmoid(z5)
						aEnd=a5

					#Compute J and add to current J
					J=J-sum(y*np.log(aEnd)+(1.0-y)*np.log(1.0-aEnd))[0]
					
					#Backward Propagation
					deltEnd=np.zeros(aEnd.shape)
					deltEnd=aEnd-y

					if hidden_layer_num==1:

						delt2=np.dot(ThetaEnd.transpose(), deltEnd) * np.vstack((1.0,sigmoidGradient(z2)))
						ThetaEnd_grad= ThetaEnd_grad +np.dot(deltEnd,a2.transpose())	
						Theta1_grad= Theta1_grad + np.dot(delt2[1:],X.reshape((1,len(X))))

					elif hidden_layer_num==2:

						delt3=np.dot(ThetaEnd.transpose(), deltEnd) * np.vstack((1.0,sigmoidGradient(z3)))
						delt2=np.dot(Theta2.transpose(), delt3[1:,:]) * np.vstack((1.0,sigmoidGradient(z2)))
						ThetaEnd_grad=ThetaEnd_grad+np.dot(deltEnd,a3.transpose())
						Theta2_grad=Theta2_grad+np.dot(delt3[1:,:],a2.transpose())					
						Theta1_grad=Theta1_grad+np.dot(delt2[1:,:],X.reshape((1,len(X))))

					else:

						delt4=np.dot(ThetaEnd.transpose(), deltEnd) * np.vstack((1.0,sigmoidGradient(z4)))
						delt3=np.dot(Theta3.transpose(), delt4[1:,:]) * np.vstack((1.0,sigmoidGradient(z3)))
						delt2=np.dot(Theta2.transpose(), delt3[1:,:]) * np.vstack((1.0,sigmoidGradient(z2)))
						ThetaEnd_grad=ThetaEnd_grad+np.dot(deltEnd, a4.transpose())
						Theta3_grad=Theta3_grad+np.dot(delt4[1:,:],a3.transpose())
						Theta2_grad=Theta2_grad+np.dot(delt3[1:,:],a2.transpose())
						Theta1_grad=Theta1_grad+np.dot(delt2[1:,:],X.reshape((1,len(X))))
						
					
					print "\b"*50+str(i)+"/"+str(m)+" rows",

				J=(1.0/m)*J
				J=J+(lamb/(2*m))*sum(nnparams**2)

				Theta1_grad=(1.0/m)*Theta1_grad
				Theta2_grad=(1.0/m)*Theta2_grad
				Theta3_grad=(1.0/m)*Theta3_grad
				ThetaEnd_grad=(1.0/m)*ThetaEnd_grad

				if hidden_layer_num==1:
					grad=np.hstack((Theta1_grad.reshape((mult_tuple(Theta1_grad.shape),)),ThetaEnd_grad.reshape((mult_tuple(ThetaEnd_grad.shape),))))

				elif hidden_layer_num==2:
					grad=np.hstack((Theta1_grad.reshape((mult_tuple(Theta1_grad.shape),)),ThetaEnd_grad.reshape((mult_tuple(ThetaEnd_grad.shape),)),Theta2_grad.reshape((mult_tuple(Theta2_grad.shape),))))
				
				else:
					grad=np.hstack((Theta1_grad.reshape((mult_tuple(Theta1_grad.shape),)),ThetaEnd_grad.reshape((mult_tuple(ThetaEnd_grad.shape),)),Theta2_grad.reshape((mult_tuple(Theta2_grad.shape),)),Theta3_grad.reshape((mult_tuple(Theta3_grad.shape),))))

				return (J,grad)

		#def gradient(nnparams,data_file,y_col,m,lamb,Theta1_par,Theta2_par,Theta3_par,ThetaEnd_par,input_nodes,hidden_layer_num, hidden_layer_nodes, output_nodes):

		self.nnparams=sp.optimize.minimize(fun=cost, x0=self.init_nnparams, jac=True, method=opt,options={'maxiter':75,'disp':True}, args=other_args)

