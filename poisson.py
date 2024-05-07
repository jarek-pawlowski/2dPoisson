import utils

poisson = utils.Poisson(grid_size=[61,61], grid_step=[1.,1.])  # grid step in nanometers
poisson.set_geometry(ny_ferr=15, ny_mono=30, nx_domain1=20, nx_domain2=40, nx_space=1)
poisson.set_epsilon()
poisson.set_boundary(positive_v=1000., negative_v=-1000.)  # in milivolts

saturation = poisson.run_solver(no_iterations=3000)

plotting = utils.Plotting(poisson, directory='./results/')
plotting.plot_potential(poisson.emp, filename='epsilon.png', label='dielectric constant')
plotting.plot_potential(poisson.fi*utils.au.Eh)
plotting.plot_potential1d(poisson.fi*utils.au.Eh, 
                          indices=[poisson.ny_mono-1, poisson.ny_mono, poisson.ny_mono+1],
                          labels=['bottom', 'middle', 'top'])
plotting.plot_saturation(saturation*utils.au.Eh)