import numpy as np
import matplotlib.pyplot as plt

def pdf(x, l_bound, r_bound, h):
    if l_bound <= x <= r_bound:
        return h
    else:
        return 0

def cdf(x, l_bound, r_bound, h):
    if x < l_bound:
        return 0
    elif l_bound <= x <= r_bound:
        return (x-l_bound)*h
    else:
        return 1

def pmf(x, l_bound, r_bound):
    y = []
    y_pmf = []
    
    for xn in x:
        if l_bound <= xn <= r_bound:
            y.append(xn)
    h = 1/len(y)

    for xn in x:
        if l_bound <= xn <= r_bound:
            y_pmf.append(h)
        else:
            y_pmf.append(0)
    
    return y_pmf

# -----------------  Question 1 ----------------------
# Item A)
n=1000
r_bound = 1
l_bound= -1

# Area under the curve = 1 (probabilities sum = 1)
# B x H = 1
# H = 1/B
h = 1/(r_bound - l_bound)


x = np.linspace(-2, 2, n)
y = np.array([pdf(xn, l_bound, r_bound, h) for xn in x])

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('PDF(x)')
plt.show()

# Item B
yCDF = np.array([cdf(xn, l_bound, r_bound, h) for xn in x])

plt.plot(x, yCDF)
plt.xlabel('x')
plt.ylabel('CDF(x)')
plt.show()

# ----------------------------------------------------

# -----------------  Question 2 ----------------------
l_bound = -2
r_bound = 2

h = 1/(r_bound - l_bound)

x = [-3,-2,-1,0,1,2,3]
y = pmf(x, l_bound, r_bound)

plt.stem(x, y)
plt.xlabel('x')
plt.ylabel('PMF(x)')
plt.show()
# ----------------------------------------------------

# -----------------  Question 4 ----------------------
