import numpy as np, sys, scipy as sp
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

	def test(self, m=None):
		if m is None:
			tries= self.m

		hx0y0=0
		hx1y0=0
		hx0y1=0
		hx1y1=0
		other=0

		for i in range(0,tries):
			if (self.logistic_hypothesis(self.X[i,:],self.theta) >=0.5 and self.y[i] ==1):
				hx1y1+=1
			elif (self.logistic_hypothesis(self.X[i,:],self.theta) <0.5 and self.y[i] ==0):
				hx0y0+=1
			elif (self.logistic_hypothesis(self.X[i,:],self.theta) >= 0.5 and self.y[i] ==0):
				hx1y0+=1
			elif (self.logistic_hypothesis(self.X[i,:],self.theta) < 0.5 and self.y[i] ==1):
				hx0y1+=1
			else:
				other+=1


		print "Accuracy on training data:\n"
		print "hx\t0\t1"
		print "y 0\t"+str(hx0y0)+"\t"+str(hx1y0)
		print "  1\t"+str(hx0y1)+"\t"+str(hx1y1)
		print "Other: "+str(other)
		print "Accuracy: %02.2f" % float((float(hx1y1+hx0y0)/float(tries))*100.)










	




		
	