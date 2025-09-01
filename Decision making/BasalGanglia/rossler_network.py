from scipy.integrate import solve_ivp
from scipy import signal
import numpy as np

class RosslerNetwork:

    def __init__(self,N, a, b, c, Iext, d, k, omega):

        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.Iext = Iext
        self.d = d
        self.k =k
        self.omega = omega


    def network(self,t,xyz):
        x = np.array(xyz[0:self.N])
        y = np.array(xyz[self.N:2*self.N])
        z = np.array(xyz[2*self.N:3*self.N])
        x_mean = np.mean(x)

        dxdt = -self.omega * y - z + self.k* 0.05 * (self.d - x_mean) + (0.5-self.k) * (0.5 * x_mean - x) 
        dydt = self.omega * x + self.a * y
        dzdt = self.b + z * (x - self.c) + self.Iext
        diff = []
        diff.extend(dxdt)
        diff.extend(dydt)
        diff.extend(dzdt)
        return np.array(diff).flatten()
    
    def run(self, time_sec = 3):
        # np.random.seed(7)
        x_intial = np.random.normal(0,0.001,self.N)       
        y_initial = np.zeros(self.N) 
        z_initial = np.zeros(self.N) 
        initial_vals = []
        initial_vals.extend(x_intial)
        initial_vals.extend(y_initial)
        initial_vals.extend(z_initial)

        t_span = (0, int(time_sec*100)) #start to end
        t_eval = np.linspace(*t_span, int(50000*time_sec)) # dt=0.002 s or 50000 samples/sec

        sol = solve_ivp(self.network, 
                t_span, 
                initial_vals, 
                t_eval=t_eval,
                vectorized=True,
                method='RK45',
                dense_output=True)
        
        return sol
