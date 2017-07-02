#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:01:56 2017

@author: aditya
"""
# intialization
import numpy as np
import os.path as op
import mne
from mne.preprocessing import ICA, create_eog_epochs
from mne import io, combine_evoked
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.time_frequency import tfr_morlet, psd_multitaper
import matplotlib.pyplot as plt
################################################################################
#preprocessing 
################################################################################


# load data 
raw = mne.io.read_raw_fif(
        "/Users/aditya/desktop/MEG_TrainingData/SelfPaced_ButtonPress/mm_selfpaced_index_rt_raw.fif",
        preload = True)


# visualize data and stim channel 
order = np.arange(raw.info['nchan'])
order[9] = 308  # We exchange the plotting order of two channels
order[308] = 9  # to show the trigger channel as the 10th channel.
#you can look at the channel by scrolling through the interactive figure
raw.plot()
raw.filter(1,50)

#filtering beta band, picking out appropriate data
picks = mne.pick_types(raw.info, meg = True, eeg = True, exclude = 'bads') # pick only MEG types

###################################################################################################
##### Artifact Correction ###########
########################################################################################
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
# when looking for artifacts manually, run below command and then hit A to enter 
# annotation mode, where using the left click and drag you can make bad segments
# and they will automatically reject those segments when epoching
raw.plot()
# for time, scroll through left and right arrow keys or bar on the
# bottom and channels are selectable through the bar on the right


######################
#### SSP Correction ###########
######################

#compute ssp projections
projs, events = compute_proj_ecg(raw, n_grad=1, n_mag=1, average=True)
print(projs)

ecg_projs = projs[-2:]
mne.viz.plot_projs_topomap(ecg_projs)

raw.info['projs'] +=  ecg_projs
evoked.add_proj(projs)
raw.plot(block=True)


######################
#### ICA Correction ###########
######################
# Again preferred method is manual annotation 
# Which is why this area is commented out, but feel free to uncomment 
# and use the ICA

#from mne.preprocessing import create_ecg_epochs
#
##pre - filtering
#raw.filter(1, 45, n_jobs=1, l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
#           filter_length='10s', phase='zero-double')
#raw.plot()
#ica = ICA(n_components=0.98, method='fastica')
#picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
#                       stim=False, exclude='bads')
#ica.fit(raw, picks=picks, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
#
## maximum number of components to reject
#n_max_ecg, n_max_eog = 3, 1  # here we don't expect horizontal EOG components
#
##detect ECG
#
#ecg_epochs = create_ecg_epochs(raw, tmin=-2, tmax=2, picks=picks)
#ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
#ica.plot_components(ecg_inds, colorbar=True)
#
#ecg_inds = ecg_inds[:n_max_ecg]
#ica.exclude += ecg_inds
#
##apply ICA
#ica.apply(raw)


# visualization of raw data
raw.plot_sensors('3d')  # in 3D


#finding events
event_id, tmin, tmax = 1, -2, 2 #specify times and event id 
events = mne.find_events(raw, stim_channel='STI101')

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
picks = mne.pick_types(raw.info, meg= True, eeg=True, stim=True, eog=False,
                       exclude='bads')

########
#Epoching
########
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=(-1.5, -1), preload=True, reject=dict(grad=4000e-13, mag=4e-12))
epochs.plot_topo_image(vmin=-200, vmax=200, title='ERF images')
epochs.plot_image(cmap='interactive') #adjust for any channel

####
#creating evoked
########
evoked = epochs.average()
mne.write_evokeds('fingerpress-ave.fif', evoked)

#Sensor data evoked visualization
evoked.plot()
evoked.plot_topomap(times = (-2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2))
evoked.plot_image()
mne.viz.plot_evoked_topo(evoked)

####
## general frequency analysis
####

# looking at frequency bands of our epochs
epochs.plot_psd()
epochs.plot_psd(fmin=2., fmax=40.)
epochs.plot_psd_topomap(ch_type='grad', normalize=True)

#PSD analysis with multitaper
f, ax = plt.subplots()
psds, freqs = psd_multitaper(epochs, fmin=2, fmax=40, n_jobs=1)
psds = 10 * np.log10(psds)
psds_mean = psds.mean(0).mean(0)
psds_std = psds.mean(0).std(0)

ax.plot(freqs, psds_mean, color='k')
ax.fill_between(freqs, psds_mean - psds_std, psds_mean + psds_std,
                color='k', alpha=.5)
ax.set(title='Multitaper PSD (gradiometers)', xlabel='Frequency',
       ylabel='Power Spectral Density (dB)')
plt.show()



###############################################################################################
# Time Frequency Analysis 
################################################################################################

#morelet frequency transformation
freqs = np.linspace(7, 50, num=40) #what frequencies are you looking for
n_cycles =  7.  # the different number of cycle per frequency?? not sure
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1) #morelet 

#####
##Time Frequency Graph
### to look at them together you may have to run these lines independently 
### working on a solution
#####

power.plot_topo(baseline=(-1.5, -1), mode='logratio', title='Average power')
power.plot([82], baseline=(-1.5, -1), mode='logratio')

fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='grad', tmin=-2, tmax=2, fmin=8, fmax=12,
                   baseline=(-1.5, -1), mode='logratio', axes=axis[0],
                   title='Alpha', vmax=0.45, show=False)
power.plot_topomap(ch_type='grad', tmin=-2, tmax=2, fmin=13, fmax=25,
                   baseline=(-1.5, -1), mode='logratio', axes=axis[1],
                   title='Beta', vmax=0.45, show=False)
mne.viz.tight_layout()
plt.show()
    
#Intertrial Power Analysis
itc.plot_topo(title='Inter-Trial coherence',  vmin=0., vmax=1., cmap='Reds')
itc.plot([82], tmin=-2, tmax=2, baseline=(-1.5,-1), mode='logratio')
itc.plot_topomap(ch_type = 'grad', tmin=-2, tmax=2, baseline = (-1.5,-1), mode = 'logratio',
                 axes=axis[0],fmin=8, fmax=12, title = 'Alpha', vmax= 0.45, show=False)
itc.plot_topomap(ch_type = 'grad',tmin=-2, tmax=2, baseline = (-1.5,-1),mode = 'logratio',
                 axes=axis[1],fmin=13, fmax=25, title = 'Beta', vmax= 0.45, show=False)
#######
########################################################################
#############################################################
#  Source Localization #
#############################################################
#######################################################################

######## 
# Compute BEM surfaces using freesurfer
# aditya$ freesurfer
# aditya$ export SUBJECTS_DIR=/Users/aditya/desktop/desktopppp/Training_MEG
# aditya$ cd $SUBJECTS_DIR
# aditya$ recon-all -i mm_mri.nii -s fingerpress -all
#######
#######
# 
#
#
# The paths to freesurfer reconstructions
subjects_dir = '/Users/aditya/desktop/MEG_TrainingData/SelfPaced_ButtonPress'
subject = 'fingerpress'

#### Plotting brain segmentations
#### After the freesurfer has done recon all, compute the BEM surfaces 
### on terminal with 'mne watershed_bem' - this will construct
### the bem folder in your directory
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', orientation='coronal')

### Getting the trans.fif file: This is the coregistration part 
### done through the terminal with mne coreg
trans = '/Users/aditya/desktop/MEG_TrainingData/SelfPaced_ButtonPress/fingerpress-trans.fif'
mne.viz.plot_trans(raw.info, trans, subject=subject, dig=True,
                   meg_sensors=True, subjects_dir=subjects_dir) ### visualizing coreg
### Computing source space 
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)

fname_cov = op.join(subjects_dir, 'fingerpress', 'fingerpress-cov.fif')
fname_cov
fname_bem = op.join(subjects_dir, 'fingerpress', 'fingerpress-bem.fif')
fname_trans = op.join(subjects_dir, 'fingerpress', 'fingerpress-trans.fif')

#visualizing BEM with source space
mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')


#### Compute Forward Solution 
conductivity = (0.3,)  # for single layer BEM
 conductivity = (0.3, 0.006, 0.3)  # for three layers

#The BEM solution requires a BEM model which describes the geometry of 
#the head the conductivities of the different tissues.
model = mne.make_bem_model(subject='fingerpress', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
mne.write_bem_solution('fingerpress-bem.fif', bem)
## Computes the forward solution
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem,
                                fname=None, meg=True, eeg=False,
                                mindist=5.0, n_jobs=2)
print(fwd)
# Computes the leadfield
leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
leadfield.shape

### Writing the forward solution
mne.write_forward_solution('fingerpress-fwd.fif', fwd )


# Making the inverse solution
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              write_inverse_operator)


## Computing the covariance
# Empty room data goes here
emptyroom = '/Users/aditya/desktop/MEG_TrainingData/SelfPaced_ButtonPress/empty_room-2.fif'
raw_empty_room = mne.io.read_raw_fif(emptyroom)
## Noise covariance compuation - can be done through either the empty room
## calculaion or using the epochs before the stimilus - I got the 
## results I showed you using the epochs method
#noise_cov = mne.compute_raw_covariance(raw_empty_room, tmin=0, tmax=None)
noise_cov = mne.compute_covariance(
    epochs, tmax=0., method=['shrunk', 'empirical'])
mne.write_cov('fingerpress-cov.fif', noise_cov)
fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)


## Pick specific types of fwd operator - not really needed
fwd = mne.pick_types_forward(fwd, meg=True , eeg=True)
info = evoked.info
### Computing inverse operator
inverse_operator = make_inverse_operator(info, fwd, noise_cov,
                                         loose=0.2, depth=0.8)
## Writing inverse operator
write_inverse_operator('fingerpress-inv.fif',
                       inverse_operator)
### Computing inverse solution
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2
stc = apply_inverse(evoked, inverse_operator, lambda2,
                    method=method, pick_ori=None)
stc.times
## Visualization of time series activation
plt.plot(1e3 * stc.times, stc.data[::100, :].T)
ts_show = -30  # show the 40 largest responses
plt.plot(stc.times,
         stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T)
plt.xlabel('time (ms)')
plt.ylabel('%s value' % method)
plt.show()
stc.times
evoked.crop()
evoked.crop(0, 0.15)
tmin, tmax = (0, 0.15)
tmax
stc_mean = stc.copy().crop(tmin, tmax).mean()
label = mne.read_labels_from_annot(subject, parc='aparc',
                                   subjects_dir=subjects_dir)[0]
aparc_label_name = 'bankssts-lh'

label = mne.read_labels_from_annot(subject, parc='aparc',
                                   subjects_dir=subjects_dir,
                                   regexp=aparc_label_name)[0]
label


stc_mean_label = stc_mean.in_label(label)
stc_mean_label
data = np.abs(stc_mean_label.data)
data

func_labels, _ = mne.stc_to_label(stc_mean_label, src=src, smooth=True,
                                  subjects_dir=subjects_dir, connected=True)
func_label = func_labels[0]
func_labels

stc_func_label = stc.in_label(func_label)
pca_func = stc.extract_label_time_course(func_label, src, mode='pca_flip')[0]



stc
stc.data[::100]
stc.data.shape
inv
from mne.viz import plot_snr_estimate
plot_snr_estimate(evoked, inverse_operator)
 stc.data[np.argsort(stc.data.max(axis=1))[ts_show:]].T
# Visualzing peak activation 
vertno_max, time_max = stc.get_peak(hemi='lh')
stc_avg.plot()
time_max
vertno_max
brain = stc.plot(surface='inflated', hemi='lh', subjects_dir=subjects_dir,
                 clim=dict(kind='value', lims=[8, 12, 15]),
                 initial_time=time_max, time_unit='s')
brain.add_foci(vertno_max, coords_as_verts=True, hemi='lh', color='blue',
               scale_factor=0.6)
brain.show_view('lateral')
brain = stc.plot(hemi='lh', subjects_dir=subjects_dir,
                 initial_time= 0.1, time_unit='s')
brain.show_view('lateral')

brain = stc.plot(hemi='both', subjects_dir=subjects_dir, initial_time=0.039000000000000146,
                 views=['ven'])


labels = mne.read_labels_from_annot('fingerpress', parc='aparc',
                                    subjects_dir=subjects_dir)
evoked = epochs.average()
evoked.crop(time_max, time_max)
time_max

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
dip.plot_locations(fname_trans, 'fingerpress', subjects_dir, mode='orthoview')

from mne.forward import make_forward_dipole
from mne.evoked import combine_evoked
from mne.simulation import simulate_evoked


fwdd, stcd = make_forward_dipole(dip, fname_bem, evoked.info, fname_trans)
pred_evokedd = simulate_evoked(fwdd, stcd, evoked.info, None, snr=np.inf)


# find time point with highes GOF to plot
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
best_idx
best_time
BEST_IDX
best_idx
# rememeber to create a subplot for the colorbar
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[10., 3.4])
vmin, vmax = -400, 400  # make sure each plot has same colour range

# first plot the topography at the time of the best fitting (single) dipole
plot_params = dict(times=best_time, ch_type='mag', outlines='skirt',
                   colorbar=False)
evoked.plot_topomap(time_format='Measured field', axes=axes[0], **plot_params)

# compare this to the predicted field
pred_evokedd.plot_topomap(time_format='Predicted field', axes=axes[1],
                         **plot_params)

# Subtract predicted from measured data (apply equal weights)
diff = combine_evoked([evoked, -pred_evokedd], weights='equal')
plot_params['colorbar'] = True
diff.plot_topomap(time_format='Difference', axes=axes[2], **plot_params)
plt.suptitle('Comparison of measured and predicted fields '
             'at {:.0f} ms'.format(best_time * 1000.), fontsize=16)


dip_fixed = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]
dip_fixed = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans,
                           pos=dip.pos[best_idx], ori=dip.ori[best_idx])[0]

dip_fixed.plot()
















