'''
main entry
'''

import numpy as np
import autograd, sys
import matplotlib.pyplot as plt

def polynomial(w, x, d=3):

   if w.size != d:
      sys.exit('wrong dim. of weights')
   d = np.arange(d) + 1
   return w[0]*x**d[0] + w[1]*x**d[1] + w[2]*x**d[2]

grad_func = autograd.grad(polynomial, 1)
X, y = np.random.rand(10, 1), np.random.rand(10, 1)
w_init = np.random.rand(3, 1)
plt.plot(np.arange(10), X, 'r-')
plt.plot(np.arange(10), [grad_func(np.array([0.5, 1, -0.5]), X[i], d=3) for i in np.arange(10)], 'g-')
plt.show()
