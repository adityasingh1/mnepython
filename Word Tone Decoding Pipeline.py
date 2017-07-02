#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:10:09 2017

@author: aditya
"""





##### Sensor Space Data Decoding
######
## Initialization
#####

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold

import mne
from mne.datasets import sample
from mne.decoding import TimeDecoding, GeneralizationAcrossTime
from mne.preprocessing import ICA

data_path = '/Users/aditya/desktop/C001/'

#Raw data initialize
raw_fnameAEF = data_path + 'C001_aef_raw_tsss.fif'
raw_fnameword = data_path + 'C001_aword_raw_tsss.fif'
event_fname = data_path + 'C001_aef_raw_tsss.fif' #????
rawAEFL = mne.io.read_raw_fif(raw_fnameAEF, preload=True) 
rawAEFR = mne.io.read_raw_fif(raw_fnameAEF, preload=True) 
rawAEF1 = mne.io.read_raw_fif(raw_fnameAEF, preload=True)
rawWORD = mne.io.read_raw_fif(raw_fnameword, preload=True)
#filt_bands = [(15, 30), (1, 40), (2,90)]
#f, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 10))
#_ = ax.plot(rawAEFL._data[0])
#for fband in filt_bands:
#    rawAEFL_filt = rawAEFL.copy()
#    rawAEFL_filt.filter(*fband, h_trans_bandwidth='auto', l_trans_bandwidth='auto',
#                    filter_length='auto', phase='zero')
#    _ = ax2.plot(rawAEFL_filt[0][0][0])
#ax2.legend(filt_bands)
#ax.set_title('Raw data')
#ax2.set_title('Band-pass filtered data')
#
#rawAEFL_band = rawAEFL.copy()
#rawAEFL_band.filter(15, 30, l_trans_bandwidth=2., h_trans_bandwidth=2.,
#                filter_length='auto', phase='zero')
#raw_hilb = rawAEFL_band.copy()
#hilb_picks = mne.pick_types(rawAEFL_band.info, meg=True, eeg=False)
#raw_hilb.apply_hilbert(hilb_picks)
#print(raw_hilb._data.dtype)
#
## Take the amplitude and phase
#rawAEFL_amp = raw_hilb.copy()
#rawAEFL_amp.apply_function(np.abs, hilb_picks)
#rawAEFL_phase = raw_hilb.copy()
#rawAEFL_phase.apply_function(np.angle, hilb_picks)
#
#f, (a1, a2) = plt.subplots(2, 1, figsize=(15, 10))
#a1.plot(rawAEFL_band._data[hilb_picks[0]])
#a1.plot(rawAEFL_amp._data[hilb_picks[0]])
#a2.plot(rawAEFL_phase._data[hilb_picks[0]])
#a1.set_title('Amplitude of frequency band')
#a2.set_title('Phase of frequency band')



# For C003 to work must drop these channels 
#rawWORD.drop_channels(ch_names = ('STI001','STI002','STI003','STI004','STI005','STI006','STI007','STI008'))
rawAEFL
#Joining the two files
rawAEFL.append([rawWORD])
rawAEFR.append([rawWORD])
#Time base line and events
tmin, tmax = -0.1, 1
event_id = dict(tone = 2, word = 1)


###Pre Processing
# Left
rawAEFL.info['bads'] += ['MEG0111', 'MEG0112','MEG0113', 'MEG0121', 'MEG0122', 'MEG0123', 'MEG0131', 'MEG0132', 'MEG0133', 'MEG0141', 'MEG0142', 'MEG0143', 'MEG0211', 'MEG0212', 'MEG0213', 'MEG0221', 'MEG0222', 'MEG0223', 'MEG0231', 'MEG0232', 'MEG0233', 'MEG0241', 'MEG0242', 'MEG0243', 'MEG0311', 'MEG0312', 'MEG0313', 'MEG0321', 'MEG0322', 'MEG0323', 'MEG0331', 'MEG0332', 'MEG0333', 'MEG0341', 'MEG0342', 'MEG0343', 'MEG0411', 'MEG0412', 'MEG0413', 'MEG0421', 'MEG0422', 'MEG0423', 'MEG0431', 'MEG0432', 'MEG0433', 'MEG0441', 'MEG0442', 'MEG0443', 'MEG0511', 'MEG0512', 'MEG0513', 'MEG0521', 'MEG0522', 'MEG0523', 'MEG0531', 'MEG0532', 'MEG0533', 'MEG0541', 'MEG0542', 'MEG0543', 'MEG0611', 'MEG0612', 'MEG0613', 'MEG0621', 'MEG0622', 'MEG0623', 'MEG0631', 'MEG0632', 'MEG0633', 'MEG0641', 'MEG0642', 'MEG0643', 'MEG0711', 'MEG0712', 'MEG0713',  'MEG0741', 'MEG0742', 'MEG0743', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823',  'MEG1011', 'MEG1012', 'MEG1013', 'MEG1511', 'MEG1512', 'MEG1513', 'MEG1521', 'MEG1522', 'MEG1523', 'MEG1531', 'MEG1532', 'MEG1533', 'MEG1541', 'MEG1542', 'MEG1543', 'MEG1611', 'MEG1612', 'MEG1613', 'MEG1621', 'MEG1622', 'MEG1623', 'MEG1631', 'MEG1632', 'MEG1633', 'MEG1641', 'MEG1642', 'MEG1643', 'MEG1711', 'MEG1712', 'MEG1713', 'MEG1721', 'MEG1722', 'MEG1723', 'MEG1731', 'MEG1732', 'MEG1733', 'MEG1741', 'MEG1742', 'MEG1743', 'MEG1811', 'MEG1812', 'MEG1813', 'MEG1821', 'MEG1822', 'MEG1823', 'MEG1831', 'MEG1832', 'MEG1833', 'MEG1841', 'MEG1842', 'MEG1843', 'MEG1911', 'MEG1912', 'MEG1913', 'MEG1921', 'MEG1922', 'MEG1923', 'MEG1931', 'MEG1932', 'MEG1933', 'MEG1941', 'MEG1942', 'MEG1943', 'MEG2011', 'MEG2012', 'MEG2013', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2041', 'MEG2042', 'MEG2043', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2141', 'MEG2142', 'MEG2143'] 
rawAEFL.filter(15, 30., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero')
#rawAEFL.filter(2, None)  # replace baselining with high-pass
#rawAEFL.plot()
############
# Right
rawAEFR.info['bads'] += ['MEG1922', 'MEG2342', 'MEG0621' , 'MEG0622', 'MEG0623', 'MEG0721', 'MEG0722', 'MEG0723', 'MEG0731', 'MEG0732', 'MEG0733', 'MEG0811', 'MEG0812', 'MEG0813', 'MEG0821', 'MEG0822', 'MEG0823', 'MEG0911', 'MEG0912', 'MEG0913', 'MEG0921', 'MEG0922', 'MEG0923', 'MEG0931', 'MEG0932', 'MEG0933', 'MEG0941', 'MEG0942', 'MEG0943', 'MEG1011', 'MEG1012', 'MEG1013', 'MEG1021', 'MEG1022', 'MEG1023', 'MEG1031', 'MEG1032', 'MEG1033', 'MEG1041', 'MEG1042', 'MEG1043', 'MEG1111', 'MEG1112', 'MEG1113', 'MEG1121', 'MEG1122', 'MEG1123', 'MEG1131', 'MEG1132', 'MEG1133', 'MEG1141', 'MEG1142', 'MEG1143', 'MEG1211', 'MEG1212', 'MEG1213', 'MEG1221', 'MEG1222', 'MEG1223', 'MEG1231', 'MEG1232', 'MEG1233', 'MEG1241', 'MEG1242', 'MEG1243', 'MEG1311', 'MEG1312', 'MEG1313', 'MEG1321', 'MEG1322', 'MEG1323', 'MEG1331', 'MEG1332', 'MEG1333', 'MEG1341', 'MEG1342', 'MEG1343', 'MEG1411', 'MEG1412', 'MEG1413', 'MEG1421', 'MEG1422', 'MEG1423', 'MEG1431', 'MEG1432', 'MEG1433', 'MEG1441', 'MEG1442', 'MEG1443', 'MEG2021', 'MEG2022', 'MEG2023', 'MEG2031', 'MEG2032', 'MEG2033', 'MEG2111', 'MEG2112', 'MEG2113', 'MEG2121', 'MEG2122', 'MEG2123', 'MEG2131', 'MEG2132', 'MEG2133', 'MEG2211', 'MEG2212', 'MEG2213', 'MEG2221', 'MEG2222', 'MEG2223', 'MEG2231', 'MEG2232', 'MEG2233', 'MEG2241', 'MEG2242', 'MEG2243', 'MEG2311', 'MEG2312', 'MEG2313', 'MEG2321', 'MEG2322', 'MEG2323', 'MEG2331', 'MEG2332', 'MEG2333', 'MEG2341', 'MEG2342', 'MEG2343', 'MEG2411', 'MEG2412', 'MEG2413', 'MEG2421','MEG2422', 'MEG2423', 'MEG2431', 'MEG2432', 'MEG2433', 'MEG2441', 'MEG2442', 'MEG2443', 'MEG2511', 'MEG2512', 'MEG2513' ,'MEG2521', 'MEG2522', 'MEG2523', 'MEG2531', 'MEG2532', 'MEG2533', 'MEG2541', 'MEG2542', 'MEG2543', 'MEG2611', 'MEG2612', 'MEG2613', 'MEG2621', 'MEG2622', 'MEG2623', 'MEG2631', 'MEG2632', 'MEG2633', 'MEG2641', 'MEG2642', 'MEG2643']  
rawAEFR.filter(15, 30., l_trans_bandwidth='auto', h_trans_bandwidth='auto',
           filter_length='auto', phase='zero')
#rawAEFR.plot()
#rawAEFR.filter(2, None)  # replace baselining with high-pass


#ICA
#picksL = mne.pick_types(rawAEFL.info, meg= True, eeg=False, stim=False, eog=False,
#                       exclude='bads') #meg can be specified to mag and grad 
#ica = ICA(n_components=0.95, method='fastica')
#ica.fit(rawAEFL, picks=picksL, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
#ica.apply(rawAEFL)
#picksR = mne.pick_types(rawAEFR.info, meg= True, eeg=False, stim=False, eog=False,
#                       exclude='bads') #meg can be specified to mag and grad
#ica = ICA(n_components=0.95, method='fastica')
#
#ica.fit(rawAEFR, picks=picksR, decim=3, reject=dict(mag=4e-12, grad=4000e-13))
#ica.apply(rawAEFR)



# Showing Events for different processes so we can classify the right
# events with different event codes
eventsAEF1 = mne.find_events(rawAEF1, stim_channel='STI101')
eventsAEF1
eventsWORD = mne.find_events(rawWORD, stim_channel='STI101')
eventsWORD


#So this shows us that there are 150 events in the first process, so we label them
# differently

eventsAEFL = mne.find_events(rawAEFL, stim_channel='STI101') #joined processes

#Changing event IDs
a = [i for i in range(len(eventsAEF1))]
eventsAEFL[a, 2] = 2  

picksL = mne.pick_types(rawAEFL.info, meg= True, eeg=False, stim=True, eog=False,
                       exclude='bads') #meg can be specified to mag and grad 

eventsAEFR = mne.find_events(rawAEFR, stim_channel='STI101') #joined processes

a = [i for i in range(len(eventsAEF1))]
a
eventsAEFR[a, 2] = 2  

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
picksR = mne.pick_types(rawAEFR.info, meg= True, eeg=False, stim=True, eog=False,
                       exclude='bads') #meg can be specified to mag and grad 


                        
                        
# Read epochs
epochsL = mne.Epochs(rawAEFL, eventsAEFL, event_id, tmin, tmax, proj=True,
                    picks=picksL, baseline=None, preload=True,
                    reject=dict(grad=4000e-13))
epochsL_list = [epochsL[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochsL_list)
data_picks = mne.pick_types(epochsL.info, meg=True, exclude='bads')

epochsR = mne.Epochs(rawAEFR, eventsAEFR, event_id, tmin, tmax, proj=True,
                    picks=picksR, baseline=None, preload=True,
                    reject=dict(grad=4000e-13))
epochsR_list = [epochsR[k] for k in event_id]
mne.epochs.equalize_epoch_counts(epochsR_list)
data_picks = mne.pick_types(epochsR.info, meg=True, exclude='bads')

epochsR.plot()
epochsL.plot()



# Temporal decoding


#####     Left Hemisphere Temporal Decoding
td = TimeDecoding(predict_mode='cross-validation', n_jobs=1)

# Fit
td.fit(epochsL)
# Compute accuracy
z = td.score(epochsL)
# Plot scores across time
td.plot(title='Left Sensor space decoding')


#####     Right Hemisphere Temporal Decoding

# Fit
td.fit(epochsR)
# Compute accuracy
x = td.score(epochsR)
# Plot scores across time
td.plot(title='Right Sensor space decoding')



############
epochsL_train = epochsL.copy().crop(tmin=-0.1, tmax=1)
labelsL = epochsL.events[:, -1]
epochsR_train = epochsR.copy().crop(tmin=-0.1, tmax=1)
labelsR = epochsR.events[:, -1]
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP
# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg='oas', log=True)
# Define a monte-carlo cross-validation generator (reduce variance):
cvL = ShuffleSplit(len(labelsL), 10, test_size=0.2, random_state=42)
scoresL = []
epochsL_data = epochsL.get_data()
epochsL_data_train = epochsL_train.get_data()
cvR = ShuffleSplit(len(labelsR), 10, test_size=0.2, random_state=42)
scoresR = []
epochsR_data = epochsR.get_data()
epochsR_data_train = epochsR_train.get_data()

# Use scikit-learn Pipeline with cross_val_score function
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scoresL = cross_val_score(clf, epochsL_data_train, labelsL, cv=cvL, n_jobs=1)
scoresR = cross_val_score(clf, epochsR_data_train, labelsR, cv=cvR, n_jobs=1)
print(scoresL.mean())
print(scoresR.mean())






from scipy import stats
mean(z[300:350])
mean(x[300:350])

mean(x[150:200])
mean(x[450:500])
mean(x[500:550])
mean(x[550:600])
mean(x[600:650])
mean(x[650:700])
mean(x[700:800])

mean(z[150:200])
mean(z[450:500])
mean(z[500:550])
mean(z[550:600])
mean(z[600:650])
mean(z[650:700])
mean(z[700:800])

stats.ttest_ind(scoresL,scoresR, equal_var = False)



epochsL_train = epochsL.copy().crop(tmin=0.1, tmax=epochsL.tmax)
labelsL = epochsL.events[:, -1] - 2
from sklearn.lda import LDA  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa
from mne.decoding import CSP
# Assemble a classifier
lda = LDA()
csp = CSP(n_components=4, reg='oas', log=True)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labelsL), 10, test_size=0.2, random_state=42)
scores = []
epochsL_data = epochsL.get_data()
epochsL_data_train = epochsL_train.get_data()

# Use scikit-learn Pipeline with cross_val_score function
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochsL_data_train, labelsL, cv=cv, n_jobs=1)


#######

## make response vector
#y = np.zeros(len(epochsL.events), dtype=int)
#y[epochsL.events[:, 2] == 1] = 2
#cv = StratifiedKFold(y=y)  # do a stratified cross-validation
#
## define the GeneralizationAcrossTime object
#gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
#                               cv=cv, scorer=roc_auc_score)
#
## fit and score
#gat.fit(epochsL, y=y)
#zg = gat.score(epochsL)
#
## let's visualize now
#gat.plot()
#gat.plot_diagonal()
############################ Right
#y = np.zeros(len(epochsR.events), dtype=int)
#y[epochsR.events[:, 2] == 1] = 2
#cv = StratifiedKFold(y=y)  # do a stratified cross-validation
#
## define the GeneralizationAcrossTime object
#gat = GeneralizationAcrossTime(predict_mode='cross-validation', n_jobs=1,
#                               cv=cv, scorer=roc_auc_score)
#
## fit and score
#gat.fit(epochsR, y=y)
#xg = gat.score(epochsR)
#
## let's visualize now
#gat.plot()
#gat.plot_diagonal()











