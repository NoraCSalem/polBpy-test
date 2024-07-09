#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# # DISPERSION ANALYSIS II
# Author: Jordan Guerra (for Villanova University). May 2024.
# 
# This tutorial illustrates the more advanced use of the package *polBpy* for performing angular dispersion analysis and studying the magnetoturbulent state of gas using dust polarimetric observations. Parameters determined with this anaylsis can be used for DCF calculations.
# 
# This tutorial uses data from literature listed [here.](https://github.com/jorgueagui/polBpy/blob/9039d4af5d25c49130bf51be7fe0ce363424edcc/refs.md)
# 
# **EXAMPLE I**: This example shows how to perform local dispersion analysis for a region of dust/gas. This analysis follows these steps: 1) calculating the autocorrelation function of the polarized flux, and measuring its width as estimation of the cloud's depth; 2) calculating a pixel-by-pixel *dispersion function* by defining a circular kernel of radius $w$ centered at each pixel; 3) fitting the *dispersion functions* with the two-scale model of Houde+09 using a Markov-Chain Montecarlo (MCMC) approach and determining the magnetotubrulent parameters. We reproduce here some results from Guerra+21, which presents uses SOFIA/HAWC+ multiwavelength polarimetric observations of the OMC-1 region.

# In[1]:


from polBpy import dispersion, fitting, utils
import numpy as np
import matplotlib.pyplot as plt
import emcee


# In this example we try to reproduce resukts from Guerra+21. Specifically, reproducing the plots in Figure 3, which utilizes the 214 $\mu$m data. First, we load all the necessary data

# In[2]:


from astropy.io import fits
import os
#cdir = os.path.dirname(os.getcwd())
file = 'Merged_Full_Polarization_Rotated.fits'
data = fits.open(file)
print(data.info())


# In[3]:


p_flux_clip = fits.open('p_flux_clip.fits')
p_flux_err_clip = fits.open('p_flux_err_clip.fits')
angles_clip = fits.open('angles_clip.fits')
angles_err_clip = fits.open('angles_err_clip.fits')


# and exclude all pixels with $p/p_{err}$ > 3.0 as done in Tutorial III

# In[4]:


p_flux = p_flux_clip[0].data # 
p_flux_err = p_flux_err_clip[0].data #
pixel_size = data[0].header['CDELT2']*3600 #in arcsec
# Find the points
m = np.where((p_flux/p_flux_err > 3.0) & (p_flux<.5))
# Create the mask
mask = np.full_like(p_flux,0.0)
mask[m] = 1.0


# Step 1: calculating $\Delta^{\prime}$ through the autocorrelation function. We call the autocorrelation function (for further details on this function see Tutorial III),

# In[5]:


Delta_p, Delta_p_err = dispersion.autocorrelation(p_flux,p_flux_err,pixsize=pixel_size,mask=mask,plots=False,hwhm=True)


# And print the $\Delta^{\prime}$ in the (multiple of) units of pixel_size (arcsec in this case)

# In[6]:


print("Delta' = %2.2f +/- %2.2f [arcmin]"%(Delta_p,Delta_p_err))


# Step 2: calcualting the *dispersion function map*. In order to calculate the map of *dispersion functions*, we need to define the relevant variables, including the beam size, and the size of the circular kernel in pixel. Following Guerra+21 we define $w$=9 pixels

# In[7]:


angles = angles_clip[0].data # 
angles_err = angles_err_clip[0].data #
beam_s = 4*pixel_size # FWHM value of the beam
win_s = 9 # pixel


# Similarly to *dispersion_function*, *dispersion_function_map* takes the polarization angles, their errors, pixel size, beam, mask, and the kernel size. In addition, you can use the keyword *verb* to printout progress of the calculation. However, the larger the size of the area to process, printing out progress can significantly slow down the process. *verb=False* only prints out which pixels are not processed.

# In[8]:


res1 = dispersion.dispersion_function_map(angles,angles_err,pixel_size,mask=mask,beam=beam_s,w=win_s,verb=False)


# For sanity check, let us visualize one dispersion function. We choose a pixel towards the center of the image

# In[9]:


epix = (35,35) # Example pixel
lvec = res1[0][epix]/3600. # \ell^2 values
disp_f = res1[1][epix] # Dispersion function
disp_f_err = res1[2][epix] # Dispersion errors
plt.errorbar(lvec,disp_f,yerr=disp_f_err,fmt='bo')
plt.xlabel(r'$\ell^{2}$ [arcmin$^{2}$]')
plt.ylabel(r'$1-\langle \cos(\Delta\phi)\rangle$')


# This *dispersion function* looks reasonable since we can see a fairly linear regime for $\sim$0.125-1 arcmin$^{2}$. 
# Step 3: fitting the *dispersion functions*. This is step is similar to that of Tutorial III, in which we fitted one function using the MCMC fitter. Here, such approach is applied to every pixel with a valid *dispersion function*. From such fits we determine the maps of paremeters. To accelerate this process, we only fit a subregion of the entire 214 $\mu$m data,

# In[10]:


m_lvec = res1[0] # in arcsec^2 (20 x 20 pixels region around the example pixel)
m_disp_f = res1[1] # dispersion function (sub)map
m_disp_f_err = res1[2] # dispersion errors (sub)map


# The function *mcmc_fit_map* uses several more keywords than *mcmc_fit* since it is implemented in a way that can be parallelized. That is, fitting processes corresponding to different pixels can be sent to different cores (this process does not require communication between the different processes). Besdides the necessary inputs (dispersion function map, their uncertainties, $\ell$ values, kernel size (in pixels), beam size (in arcsecs), we can provide initial-guess values for the parameters $a_{2}$, $\delta$, and $\Delta^{\prime}/[\langle B_{t}^{2}\rangle/\langle B_{0}^{2}\rangle]$. Keyword *num* sets the number of random walkers and iterations for the MCMC fittting and its default value is 500. Keyword *n_cores* allows you to select the number of cores/processes to be executed and its default value (*n_cores=False*) is half of available cores.

# In[ ]:


res2 = fitting.mcmc_fit_map(m_disp_f,m_lvec,m_disp_f_err,win_s,pixel_size,beam=beam_s,a2=3.855E-02,delta=39.56,f=576.37,
                            num=600,verb=False,n_cores=8)


# The output of *mcmc_fit_map* is a dictionary with maps of all three parameters mentioned above, plus maps of goodness-of-fit $\chi$ and non-linear rank correlation coefficient $\rho$.
# 
# Let us inspect the fit for the example pixel plotted above,

# In[12]:


"""
DO NOT RERUN
"""

l = np.arange(0,10000) # define some values of \ell^2 in arcsec for the model
# Choose the same pixel example
# Evaluate the two-scale function
nepix = (10,10) # since it will correspond to epix in our sub-array
a2 = res2['a'][nepix]# arcsec^-2
delta = res2['d'][nepix] # arcsec
ratio = res2['f'][nepix] # arcsec
f = fitting.model_funct(l,a2,delta,ratio,beam=beam_s)


# Replot the dispersion function and fit,

# In[ ]:


"""
DO NOT RERUN
"""


lvec = res1[0][epix]/3600.
disp_f = res1[1][epix]
disp_f_err = res1[2][epix]
plt.figure(figsize=(7,6))
plt.errorbar(lvec,disp_f,yerr=res2['chi'][10,10]*disp_f_err,fmt='bo') #blue points, fitting 
plt.xlabel(r'$\ell^{2}$ [arcmin$^{2}$]')
plt.ylabel(r'$1-\langle \cos(\Delta\phi)\rangle$')
lratio = r'$\Delta^{\prime}/[\langle B_{t}^{2}\rangle/\langle B_{0}^{2}\rangle]$'
label = r'Two-scale model with: $a_{2}$=%2.2f, $\delta$=%2.2f, %s=%2.2f'%(a2*3600.,delta,lratio,ratio)
plt.plot(l/3600., f,c='red',label=label) #red fit, 
plt.ylim([0.,0.3])
plt.xlim([0.,10.])
plt.legend()


# We can see from this plot that the MCMC solver finds parameter values that describe the data well. Better parameters can be obtained by giving *mcmc_fit_map* different initial-guess values or a larger number of walker/iterations. Good initial-guess values are those obtained by the dispersion analysis performed to the entire region, such as described in Tutorial III.
# 
# For consistency to values reported in Guerra+21, we print the parameters fro this fit

# In[14]:


a2 = res2['a'][nepix]*3600.*1000 # Units of 10^{-3} arcmin^{-2}
print("a_2 = %2.3E [arcmin^-2]"%a2)


# In[15]:


print("delta = %2.2f [arcsec]"%delta)


# In[16]:


Delta_p *= 60. # Cloud's depth in arcsec
ratio = Delta_p/res2['f'][nepix]
print("ratio = %2.2f"%ratio)


# We can now visualize the maps of parameters calculated,

# In[17]:


a2_map = res2['a']*3600*1000 # Units of 10^{-3} arcmin^{-2}
delta_map = res2['d'] # arcsec
ratio_map = Delta_p/res2['f']
plt.figure(figsize=(20,5))
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,nrows=1)
g = ax1.imshow(a2_map,origin='lower',vmin=1.,vmax=100.)
plt.colorbar(g,ax=ax1,orientation='horizontal',label=r'[10$^{-3}$ arcmin$^{-2}$]')
ax1.set_title(r'$a_{2}$')
g = ax2.imshow(delta_map,origin='lower',vmin=0.,vmax=50.)
plt.colorbar(g,ax=ax2,orientation='horizontal', label='[arcsec]')
ax2.set_title(r'$\delta$')
g = ax3.imshow(ratio_map,origin='lower',vmin=0.,vmax=1.)
plt.colorbar(g,ax=ax3,orientation='horizontal',label='')
bratio = r'$\langle B_{t}^{2}\rangle/\langle B_{0}^{2}\rangle$'
ax3.set_title(bratio)


# The MCMC solver sometimes provides solutions for pixels that are out-of-statistics for their inmediate region. See for example the value ~100 in the $\a_{2}$ around the pixel (10,15). Therefore, we can "clean" the maps by performing a $\sigma$ clipping. That is, each pixel value is compared to the statistics in a 3x3 pixels kernel aroudn it. If the value is outside the range $median \pm n\sigma$ then the value for the such pixel is interpolated from its nearest neighbors.
# 
# In this case we choose $n=2$,

# In[ ]:


c_a2_map = utils.clean_map(a2_map,sigma=2,w=3)
c_delta_map = utils.clean_map(delta_map,sigma=2,w=3)
c_ratio_map = utils.clean_map(ratio_map,sigma=2,w=3)


# And visualize the "clean" maps,

# In[ ]:


plt.figure(figsize=(20,5))
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,nrows=1)
g = ax1.imshow(c_a2_map,origin='lower',vmin=1.,vmax=100.)
plt.colorbar(g,ax=ax1,orientation='horizontal',label=r'[10$^{-3}$ arcmin$^{-2}$]')
ax1.set_title(r'$a_{2}$')
g = ax2.imshow(c_delta_map,origin='lower',vmin=0.,vmax=50.)
plt.colorbar(g,ax=ax2,orientation='horizontal', label='[arcsec]')
ax2.set_title(r'$\delta$')
g = ax3.imshow(c_ratio_map,origin='lower',vmin=0.,vmax=1.)
plt.colorbar(g,ax=ax3,orientation='horizontal',label='')
bratio = r'$\langle B_{t}^{2}\rangle/\langle B_{0}^{2}\rangle$'
ax3.set_title(bratio)
fig.suptitle('Clean Maps')


# We see in these clean maps that the large value in the middle of the $a_{2}$ map dissapeared. The overall spatial distribution of the maps appears "smoother". However, changing the value of $\sigma$ or $w$ can help make these maps even smoother.
# 
# We now save these maps into a pickle file for using in Tutorial V where we will use the $bratio$ map for computing a map of $B_{\rm POS}$ using the DCF approximations,

# In[ ]:


res3 = {'a2_map':c_a2_map,'delta_map':c_delta_map,'ratio_map':c_ratio_map}
import pickle
pickle.dump(res3,open(cdir+'/data/disp_analysis_maps.pk',"wb"))


# In[ ]:





# In[ ]:





# In[ ]:




