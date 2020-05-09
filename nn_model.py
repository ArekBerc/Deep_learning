import numpy as np
import scipy

def sigmoid(Z):    
	A = 1/(1+np.exp(-Z))
	cache = Z

	return A, cache
def tanh(Z):
	A = np.tanh(Z)
	cache = Z
	return A,cache
def relu(Z):
	A = np.maximum(0,Z)

	assert(A.shape == Z.shape)

	cache = Z 
	return A, cache
def relu_backward(dA, Z):
	dZ = np.array(dA, copy=True)  
	dZ[Z <= 0] = 0

	assert (dZ.shape == Z.shape)

	return dZ
def sigmoid_backward(dA, Z):
	x = 1/(1+np.exp(-Z))
	dZ = dA * x * (1-x)

	assert (dZ.shape == Z.shape)

	return dZ
def tanh_backward(dA, Z):
	x = np.tanh(Z)
	dZ = dA * (1 - np.pow(x,2))

	return dZ
def parameters(layer_dims):
	params = {}
	L = len(layer_dims)

	for l in range(1,L):
		params['W'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
		params['b'+str(l)] = np.zeros((layer_dims[l],1))

		assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
		assert(params['b' + str(l)].shape == (layer_dims[l], 1))

	return params

def forward_lin(A,W,b):
	Z = np.dot(W,A) + b
	print("g")
	assert(Z.shape == (W.shape[0], A.shape[1]))

	stash = (A,W,b)

	return Z,stash

def forward_lin_a(A_p,W,b,activation):
	if activation == "sigmoid":
		print("c")
		Z,lin_cache = forward_lin(A_p,W,b)
		A, active_cache = sigmoid(Z)

	elif activation == "relu":
		print("d")	
		Z,lin_cache = forward_lin(A_p,W,b)
		print("e")
		A,activation_cache = relu(Z)
		print("f")
	assert (A.shape == (W.shape[0], A_p.shape[1]))
	cache = (lin_cache,active_cache)
	return A,cache


def forward_model(X,params):
	caches =[]
	A = X
	L=len(params) // 2

	for l in range(1,L):
		A_p = A
		print("b")
		A,cache = forward_lin_a(A_p,params['W' + str(l)],params['b'+ str(l)],"relu")
		

		caches.append(cache)

	AL,cache = forward_lin_a(A,params['W' + str(L)],params['b' + str(L)],"sigmoid")
	caches.append(cache)
	assert(AL.shape == (1,X.shape[1]))
	return AL,caches


def cost_function(AL,Y):

	m = Y.shape[1]
	cost = np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)) /(-m)
	cost = np.squeeze(cost)
	assert(cost.shape == ())
	return cost


def linear_backward(dZ,cache):
	A_prev,W,b = cache
	m = A_prev.shape[1]


	dW = np.dot(dZ,A_prev.T) / m
	db = np.sum(dZ,axis = 1,keepdims = True) / m
	dA_prev = np.dot(W.T,dZ)
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev,dW,db

def backward_lin_a(dA,cache,activation):
	linear_cache,activation_cache = cache

	if activation == "relu":
		dZ= relu_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA,activation_cache)
		dA_prev,dW,db = linear_backward(dZ,linear_cache)


	return dA_prev,dW,db

def backward_model(AL,Y,caches):
	grads={}
	L = len(caches)
	m = AL.shape[1]
	Y = Y.reshape(AL.shape)

	dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_lin_a(dAL, current_cache, "sigmoid")	
	for l in reversed(range(L-1)):
		current_cache = caches[l]
		dA_prev_temp, dW_temp, db_temp = backward_lin_a(grads["dA" + str(l+1)], current_cache, "relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return	grads

def update_params(params,grads,learning_rate):
	L = len(params)

	for l in range(L):
		params["W" + str(l+1)] = parameters["W" + str(l+1)] - (learning_rate * grads["dW" + str(l+1)])
		params["b" + str(l+1)] = parameters["b" + str(l+1)] -(learning_rate * grads["db" + str(l+1)])

	return params

def model_NN(X,Y,layer_dims,learning_rate = 0.0070,num_iterations =3200,print_cost = True):
	costs = []
	params = parameters(layer_dims)

	for i in range(0,num_iterations):
		AL,caches = forward_model(X,params)
		print("a")
		cost = cost_function(AL,Y)
		grads = backward_model(AL,Y,caches)
		params = update_params(params,grads,learning_rate)
		
		if print_cost and i % 100 == 0:
			costs.append(cost)
			print ("Cost: %f" %(cost))

	return params