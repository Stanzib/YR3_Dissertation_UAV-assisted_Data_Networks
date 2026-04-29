import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as smp
print("" \
"")
#plt.style.use(['seaborn-v0_8','notebook'])

x = np.linspace(0,3,100) # 100 things between 0 and 3
f = 2*np.exp(-2*x) # little f
F = 1-np.exp(-2*x) # big F

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.plot(x,F, label=r'$F(x)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.title("f(x) and CDF of f(x) = F(x)")
#plt.show()

# using random numbers to evaluate F^-1(U) which should be distrabuted acourding to f(x)


Us = np.random.rand(10000)
F_inv_Us = -np.log(1-Us)/2

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(F_inv_Us, histtype='step', color='red', density='norm', bins=100, label='$F^{-1}(u)$')
# Normalised under the histogram so that area equals one!!
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
#plt.show()


############ What if f(x) isnt invertable mathmatically #################

# using sympy to define a distrabution 
x, y, F1, F2, E1, E2 = smp.symbols('x y F_1 F_2 E_1 E_2', real=True, positive=True)
fs = F1*smp.exp(-smp.sqrt(x/E1)) + F2*smp.exp(-smp.sqrt(x/E2)) # the s means symblic
print(fs)

Fs = smp.integrate(fs, (x,0,y)).doit() # intergrating x from 0 to y (cant use x again therefore use y)
print(Fs)

# now need to take inverse. to do this we will give it numbers and it will return numbers

Fn = smp.lambdify((y, E1, E2, F1, F2), Fs)
fn = smp.lambdify((x, E1, E2, F1, F2), fs)

E1 = E2 = 0.2
F1 = 1.3
F2 = 1.4
x = np.linspace(0,5,1000)
f = fn(x, E1, E2, F1, F2) # converting from sympy to be used by numpy
F = Fn(x, E1, E2, F1, F2)

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.plot(x,F, label=r'$F(x)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
#plt.show()

F_inv_Us = x[np.searchsorted(F[:-1], Us)] # inverting it
# looking at array of values of x at all the different indercies and sorting them
# give x value such that y(from Fs) is 0.6

plt.figure(figsize=(8,3))
plt.plot(x, f, label=r'$f(x)$')
plt.hist(F_inv_Us, histtype='step', color='red', density='norm', bins=100, label='$F^{-1}(u)$')
plt.legend()
plt.xlabel('$x$', fontsize=20)
plt.legend()
plt.xlim(0,2)
#plt.show()


print("" \
"")