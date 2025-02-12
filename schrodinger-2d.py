#!/usr/bin/env python
# coding: utf-8

# In[34]:


from dataclasses import dataclass
import numpy as np
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt
import matplotlib.animation as animation
# get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:


@dataclass
class Vector:
    x: float
    y: float
    t: float = -1


# ### Units

# In[3]:


hbar = 1.0 # reduced
mass = 1.0 # natural/quantum units


# ### Grid

# In[4]:


grid_size     = Vector(200, 200) # px
physical_size = Vector(10,  10)
grid_range = [-physical_size.x/2, physical_size.x/2, 
              -physical_size.y/2, physical_size.y/2]

xvalues, yvalues = np.meshgrid(np.linspace(grid_range[0], grid_range[1], grid_size.x), 
                               np.linspace(grid_range[2], grid_range[3], grid_size.x), 
                               indexing='ij')


# ### Barrier

# In[5]:


barrier_x = int(grid_size.x / 2)
slit_width = 5 # px
slit_sep = 40
barrier_thickness = 4
barrier_potential = 1e6


# Set the entire field to be zero, then set the potential energy barriers get "infinite" (really high) potential energy (thus blocking the wave function from propagating across it, and instead, it diffuses). Then, we can define slits by setting specific areas of the potential field to be zero again.

# In[6]:


potential_field = np.zeros((grid_size.x, grid_size.y))
# potential_field[barrier_x - barrier_thickness:barrier_x + barrier_thickness, :] = barrier_potential

# Add slits along this vertical barrier
slit1_y = grid_size.y // 2 - slit_sep // 2
slit2_y = grid_size.y // 2 + slit_sep // 2


# ### Wave Function

# In[7]:


initial_position = Vector(0, 0)
momentum = Vector(0, 5)
packet_width     = 1.0


# Wave packet general formula (for a free atom) is
# 
# $$\psi(x,y,t) = \frac{1}{\sigma \sqrt{\pi}}\exp(-\frac{(x-x_{0}-v_{x}t)^2 + (y-y_{0}-v_{y}t)^2}{2\sigma^2})\exp(i(k_{x} x + k_{y} y - \omega))$$
# 
# where
# - $(x_{0}, y_{0})$ is the initial center of the wave packet.
# - $\sigma$ is the spread (or width) of the wave packet.
# - $k_x, k_y$ are the initial wave numbers (related to momentum via $p = \hbar k)$. 2-D wave number is $k = \sqrt{k_{x}^{2} + k_{y}^{2}}$.
# - $v_{x} = \frac{\hbar k_{x}}{m}$ and $v_{y} = \frac{\hbar k_{y}}{m}$ are the group velocities in the $x$ and $y$ directions.
# - $\omega = \frac{\hbar k^2}{2m}$
# - The first exponential term makes the wave function Gaussian in space.
# - The second exponential term gives it an initial momentum.

# In[8]:


wave_number = Vector(momentum.x / hbar, momentum.y / hbar)
wave_number_combined = np.sqrt(wave_number.x ** 2 + wave_number.y ** 2)
group_velocity = Vector(hbar * wave_number.x / mass, hbar * wave_number.y / mass)
dispersion_relation = (hbar * wave_number_combined ** 2) / (2 * mass)


# In[9]:


pre_factor = 1/(packet_width * np.sqrt(np.pi))
phase = lambda x,y,t: wave_number.x * x + wave_number.y * y - dispersion_relation * t
envelope = lambda x,y,t: np.exp(-((x - initial_position.x) ** 2 + (y - initial_position.y) ** 2) / (2 * packet_width ** 2))
psi_function = lambda x,y,t: pre_factor * envelope(x,y,t) * np.exp(1j * phase(x,y,t))


# ### Laplacian and the Hamiltonian

# $$\hat{H} = -\frac{\hbar^{2}}{2m} \triangledown^{2}$$
# and
# $$\triangledown^{2} = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$$
# for a free particle in 2 dimensions.
# 
# For the laplacian, we can estimate it with finite differences:
# $$\frac{\partial^2}{\partial x^2} = \frac{\psi(x + \delta x, y, t) - 2 \psi(x,y,t) + \psi(x - \delta x, y, t)}{\delta x^2},$$
# and vice-versa for $y$.

# In[10]:


second_derivative_x = lambda psi, dx: (lambda x, y, t: (psi(x + dx, y, t) - 2 * psi(x, y, t) + psi(x - dx, y, t))/(dx ** 2))
second_derivative_y = lambda psi, dy: (lambda x, y, t: (psi(x, y + dy, t) - 2 * psi(x, y, t) + psi(x, y - dy, t))/(dy ** 2))


# In[11]:


laplacian = lambda psi, dx, dy: (lambda x, y, t: second_derivative_x(psi, dx)(x,y,t) + 
                                         second_derivative_y(psi, dy)(x,y,t))
hamiltonian_constant = -(hbar ** 2)/(2 * mass)
hamiltonian = lambda psi, dx, dy: (lambda x, y, t: hamiltonian_constant * laplacian(psi, dx, dy)(x,y,t))


# ### Euler Simulation

# The euler equation can be written as
# 
# $$\psi(x, y, t + \delta t) = \psi(x,y,t) + \delta t \left( -\frac{i}{\hbar} \hat{H} \psi(x,y,t) \right)$$
# 
# because by the time-dependent Schrodinger equation
# 
# $$i\hbar \frac{\partial \psi}{\partial t} = \hat{H} \psi$$

# In[12]:


schrodinger_euler = lambda psi, dx, dy, dt: (lambda x, y, t: dt * (1j / hbar) * hamiltonian(psi, dx, dy)(x,y,t))


# In[13]:


def perform_euler_simulation(psi, initial_time, final_time, dt):
    iterations = int((final_time - initial_time) / dt)

    dx = physical_size.x / grid_size.x
    dy = physical_size.y / grid_size.y

    psi_data = []
    time_data = []
    
    time = initial_time
    psi_data.append(psi(xvalues, yvalues, 0))

    calc_pdf = lambda psi: np.sum(np.abs(psi) ** 2) * dx * dy
    pdf_data = [ calc_pdf(psi_data[0]) ]
    
    for i in range(iterations):
        time += dt
        time_data.append(time)
        psi_next = psi_data[i] + schrodinger_euler(psi, dx, dy, dt)(xvalues, yvalues, time)
        pdf = calc_pdf(psi_next)

        # renormalize psi
        psi_next /= np.sqrt(pdf)

        psi_data.append(psi_next)
        pdf_data.append(pdf)
        
    # Verify that psi is normalized
    is_normalized = np.allclose(pdf_data, 1.0, atol=1e-2)

    if is_normalized:
        print('psi is normalized.')
    else:
        print(f'psi is NOT normalized, mean {np.mean(pdf_data)}, first 10: {pdf_data[0:10]}')

    return time_data, psi_data, pdf_data


# ### Solve and Graph

# In[39]:


def animate_psi(time_data, psi_data):
    fig, ax = plt.subplots()

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    potential_display = np.ma.masked_where(potential_field == 0, potential_field)
    ax.contour(xvalues, yvalues, potential_field, levels=[barrier_potential / 2], colors='white', linewidths=1.5)
    heatmap_anim = ax.imshow(np.abs(psi_data[0])**2, extent=grid_range, origin='lower', cmap='inferno', alpha=0.7, vmin=0, vmax=0.3)

    cbar = plt.colorbar(heatmap_anim, ax=ax, label='Probability Density')
    title = ax.set_title(f"Time-Dependent Euler Simulation (t={time_data[0]} s)")

    def update(frame):
        psi_abs2 = np.abs(psi_data[frame])**2
        heatmap_anim.set_data(psi_abs2)

        # heatmap_anim.set_clim(vmin=np.min(psi_abs2), vmax=np.max(psi_abs2))

        title.set_text(f"Time-Dependent Euler Simulation (t={time_data[frame]:.2f} s)")
        return heatmap_anim, title

    ani = animation.FuncAnimation(fig, update, frames=len(time_data), blit=False, interval=50)

    plt.show()


# In[40]:


time_data, psi_data, pdf_data = perform_euler_simulation(psi_function, 
                                               initial_time=0,
                                               final_time=2,
                                               dt=0.001)
animate_psi(time_data, psi_data)

plt.figure()
plt.plot(time_data, pdf_data[:-1])
plt.show()
