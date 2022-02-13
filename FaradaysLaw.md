## Year 1 Experiment: Investigating the Faraday Effect

# Efficiency calculations

The following below is my code for calculating the efficiency of the transformers. 
This code uses the data set for the air core. 

```markdown
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

data = np.loadtxt("DataAir.txt", skiprows = 1)
P_P = data[:,0]
sigma_P = data[:,1]
P_S = data[:,2]
sigma_S = data[:,3]

fit,cov = np.polyfit(P_P,P_S,1,cov=True)

sig_0 = np.sqrt(cov[0,0]) #The uncertainty in the slope
sig_1 = np.sqrt(cov[1,1]) #The uncertainty in the intercept

print('Slope = %.3e +/- %.3e' %(fit[0],sig_0))# Note the %.3e forces the values to be printed in scientific notation with 3 decimal places.
print('Intercept = %.3e +/- %.3e' %(fit[1],sig_1))

Fit = np.poly1d(fit)
plt.xlabel("Power Primary")
plt.ylabel("Power Secondary")
plt.title("Efficiency")
plt.errorbar(P_P, P_S,xerr = sigma_P, yerr=sigma_S, fmt='o', mew=2, ms=3, capsize=4)
plt.plot(P_P,Fit(P_P))
plt.grid()
plt.show()
```
This was the image for our graph of an air core. 

![image](https://user-images.githubusercontent.com/99592215/153750492-8e75af6a-becd-4460-b6d4-9bbd57631b4e.png)

This was the graph for the Ferite core. 

![image](https://user-images.githubusercontent.com/99592215/153750526-6db9c7a9-0bdd-4995-9bcd-fa417f957057.png)


# The Simulation

The following was or code for the simulation:

```markdown
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

NP = 100
NS = 200
InputFreq = 10
InputAmplitude = 10
Efficiency = 0.9

Px = np.arange(0,np.pi,0.01)   # start,stop,step
Py = InputAmplitude*np.sin(Px*InputFreq)

Vs = Efficiency*InputAmplitude*NS/NP

Sx = np.arange(0,np.pi,0.01) 
Sy = Vs*np.sin(Sx*InputFreq)

plt.xlabel("Time / t")
plt.ylabel("Voltage / V")
plt.title("Simulation")
plt.plot(Sx,Sy, label="Secondary Voltage")

plt.plot(Px,Py, label="Primary Voltage")

plt.legend(loc="upper right")
plt.grid()
plt.show()
```

With the parameters in the code above, the following graph was produced:


![image](https://user-images.githubusercontent.com/99592215/153750606-6d9aab6a-b18d-4166-aefe-f7fe24850d9a.png)


