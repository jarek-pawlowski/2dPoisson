import os
import numpy as np
import matplotlib.pyplot as plt

def abs2(x):
    return x.real**2 + x.imag**2

class AtomicUnits:
    """Class storing atomic units.

    All variables, arrays in simulations are in atomic units.

    Attributes
    ----------
    Eh : float
        Hartree energy (in meV)
    Ah : float
        Bohr radius (in nanometers)
    Th : float
        time (in picoseconds)
    Bh : float
        magnetic induction (in Teslas)
    """
    # atomic units
    Eh=27211.4 # meV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()

# dielectric constants 
ehbn = 3.6
ews2 = 6.3
eferr = 1.0  # ?

class Poisson:
    
    def __init__(self, grid_size=[61,51], grid_step=[1.,1.]):
        self.nx = grid_size[0]
        self.ny = grid_size[1]
        self.dx = grid_step[0]/au.Ah
        self.dy = grid_step[1]/au.Ah
        self.fi = np.zeros((self.nx,self.ny))
        self.fi1 = np.zeros_like(self.fi)
        self.ro = np.zeros_like(self.fi)
        self.emp = np.ones((self.nx,self.ny))
        
    def set_geometry(self, ny_ferr, ny_mono, nx_domain1, nx_domain2, nx_space): 
        # geometry scheme    
        # ------------------------------------
        # hbn
        # monolayer                             ny_mono
        # hbn    
        # ++++++++++   ----------   ++++++++++  ny_ferr
        # eferr 
        # ----------   ++++++++++   ----------  0
        #            |            |
        #       nx_domain1    nx_domain2  (space between domains = nx_space*2+1)
        #
        self.ny_ferr = ny_ferr
        self.ny_mono = ny_mono
        self.nx_domain1 = nx_domain1
        self.nx_domain2 = nx_domain2
        self.nx_space = nx_space
        
    def set_epsilon(self):
        self.emp[:,:self.ny_ferr] = eferr  # for y < nz_ferr
        self.emp[:,self.ny_ferr:] = ehbn  # for y >= nz_ferr
        self.emp[:,self.ny_mono] = ews2  # for y == nz_mono
        
    def set_boundary(self, positive_v, negative_v):
        self.pos_v = positive_v/au.Eh 
        self.neg_v = negative_v/au.Eh 
        
    def apply_boundary(self, fi):
        # lower layer
        fi[:self.nx_domain1-self.nx_space,0] = self.neg_v 
        fi[self.nx_domain1+self.nx_space:self.nx_domain2-self.nx_space,0] = self.pos_v
        fi[self.nx_domain2+self.nx_space:,0] = self.neg_v
        # upper layer
        fi[:self.nx_domain1-self.nx_space,self.ny_ferr] = self.pos_v 
        fi[self.nx_domain1+self.nx_space:self.nx_domain2-self.nx_space,self.ny_ferr] = self.neg_v
        fi[self.nx_domain2+self.nx_space:,self.ny_ferr] = self.pos_v
        
    def run_solver(self, no_iterations=50):
        ddx=self.dx**2
        ddy=self.dy**2
        collect_cental_fis = []  # to control whether we reach saturation of potential values
        for it in range(no_iterations):
            for i in range(1,self.nx-1):
                for j in range(1,self.ny-1):        
                    w1=(self.fi[i-1,j]+self.fi[i+1,j])/ddx+(self.fi[i,j-1]+self.fi[i,j+1])/ddy
                    w2=(self.emp[i+1,j]-self.emp[i-1,j])*(self.fi[i+1,j]-self.fi[i-1,j])/ddx+(self.emp[i,j+1]-self.emp[i,j-1])*(self.fi[i,j+1]-self.fi[i,j-1])/ddy
                    self.fi1[i,j]=w1+w2/self.emp[i,j]/4.
            self.fi1 -= 4.*np.pi*self.ro/self.dx/self.dy/self.emp # in our case rho is zero everywhere
            self.fi1 /= 2./ddx+2./ddy  
            # apply boundary conditions at "gates"
            self.apply_boundary(self.fi1)
            # apply boundary conditions at computational box edges
            # we assume charge-neutrality and thus zeroing normal component of the E-field
            self.fi1[0,:] = self.fi[1,:]
            self.fi1[-1,:] = self.fi[-2,:]
            #self.fi1[:,-1] = self.fi[:,-2]
            self.fi1[:,-1] = 0.
            # updating
            self.fi = self.fi1*.9+self.fi*.1
            collect_cental_fis.append(self.fi[int(self.nx/2), int(self.ny/2)])
        return np.array(collect_cental_fis)
            
class Plotting:
    def __init__(self, solver, directory=None):
        self.solver = solver
        if directory is not None:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'
        self.pointsize = 2.
        
    def plot_potential(self, fi, filename='potential.png', label='potential (mV)'):
        fig, ax = plt.subplots()
        ax.set_aspect(1.)
        ax.set_xlabel(r'$x$ (nm)')
        ax.set_ylabel(r'$y$ (nm)')
        X,Y = np.meshgrid(np.linspace(0, (self.solver.nx-1)*self.solver.dx*au.Ah, num=self.solver.nx, endpoint=True),
                          np.linspace(0, (self.solver.ny-1)*self.solver.dy*au.Ah, num=self.solver.ny, endpoint=True))
        potential = ax.pcolormesh(X,Y,fi.T, cmap='coolwarm')
        cbar = fig.colorbar(potential, ax=ax)
        cbar.set_label(label)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_potential1d(self, fi, indices, labels, filename='potential1d.png'):
        _, ax = plt.subplots()
        ax.set_xlabel(r'$x$ (nm)')
        ax.set_ylabel(r'$\phi$ (mV)')
        X = np.linspace(0, (self.solver.nx-1)*self.solver.dx*au.Ah, num=self.solver.nx, endpoint=True)
        for idx,label in zip(indices, labels):
            ax.plot(X, fi[:,idx], label=label)
        ax.legend()
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
    def plot_saturation(self, values, filename='saturation.png'):
        _, ax = plt.subplots()
        ax.set_xlabel(r'subsequent iterations')
        ax.set_ylabel(r'$\phi$ (mV)')
        ax.plot(range(len(values)), values)
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()
        
        


