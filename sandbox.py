import numpy as np


n_options = 8
probs = np.array([.75,.25,.75,.25,.75,.25,.75,.25])
rmag = 	np.array([.5,.5,.5,.5,.0,.0,.0,.0])
lmag = 	np.array([.0,.0,.0,.0,-.5,-.5,-.5,-.5])
ctx = np.array([0,0,1,1,2,2,3,3])

full_info = [True,True,False,False,True,True,False,False]	# full or partial info
order = np.repeat([0,1,2,3],24)								# order of ctx types
np.random.shuffle(order)
print(order)
