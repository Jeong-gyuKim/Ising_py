import numpy as np
#log(a+b) = log(a)+log(1+b/a) = log(a)+log(1+exp(log(b)-log(a)))

Z = 0
z=0

a,b = np.random.rand(2,10)
for i in range(10):
    
    Z += np.log(1 + np.exp(np.log(a[i])+b[i] - Z))
    z+=a[i]*np.exp(b[i])
Z += np.log(1 - np.exp(-Z))
print((np.exp(Z)) - (z))