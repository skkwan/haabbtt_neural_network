import ROOT
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Activation, Dense, Dropout

import joblib

import uproot

import time


t_before = time.perf_counter()

# 1bNN
tload_before = time.perf_counter()
savedscaler = joblib.load('trained_models_rerunBtag/mt1b_scaler.gz')
savedmodel = keras.models.load_model('trained_models_rerunBtag/mt1b_model')
savedmodel.summary()
tload_after = time.perf_counter()
print(">>> Loaded model in {} seconds!".format(tload_after- tload_before))


#2018
infolder = 'test_input/'
outfolder = 'test_output/'

inpaths  = ['postprocessed_ntuple_VBFHToTauTau.root']
outpaths = ['dnn_VBFHToTauTau.root']


tRDF_before =time.perf_counter() 
for i in range(len(inpaths)):
    inpaths[i] = infolder + inpaths[i]
    outpaths[i] = outfolder + outpaths[i]
    
allfiles = []
for i in range(len(inpaths)):
    allfiles.append(ROOT.RDataFrame('mutau/event_tree', inpaths[i]))

# Nominal 1bNN 2018 = A -> B1 -> C1 -> D1 -> H -> Clear cells and restart notebook

# Nominal variables for 1bNN
mymu = 'ROOT::Math::PtEtaPhiMVector(pt_1_nominal, eta_1, phi_1, m_1_nominal)'
mytau = 'ROOT::Math::PtEtaPhiMVector(pt_2_nominal, eta_2, phi_2, m_2_nominal)'
mymet = 'ROOT::Math::PtEtaPhiMVector(met_nominal, 0, metphi_nominal, 0)'
# Use m_sv
# mytt = 'ROOT::Math::PtEtaPhiMVector((mymu+mytau+mymet).Pt(),(mymu+mytau+mymet).Eta(),(mymu+mytau+mymet).Phi(),m_sv)'
mytt = 'mymu+mytau'#for the moment take only visible parts to reconstruct the ditau system
myb1 = 'ROOT::Math::PtEtaPhiMVector(bpt_deepflavour_1,beta_deepflavour_1,bphi_deepflavour_1,bm_deepflavour_1)'
m_btt = '(mytt+myb1).M()'
m_b1mu = '(mymu+myb1).M()'
m_b1tau = '(mytau+myb1).M()'
dR_tt = 'ROOT::Math::VectorUtil::DeltaR(mymu,mytau)'
dR_b1mu = 'ROOT::Math::VectorUtil::DeltaR(mymu,myb1)'
dR_b1tau = 'ROOT::Math::VectorUtil::DeltaR(mytau,myb1)'
dR_b1tt = 'ROOT::Math::VectorUtil::DeltaR(myb1,mytt)'
tt_pt = 'mytt.Pt()'
tt_eta = 'mytt.Eta()'

mT_mu = 'sqrt(pow(mymu.Pt()+mymet.Pt(),2)-pow(mymu.Px()+mymet.Px(),2)-pow(mymu.Py()+mymet.Py(),2))'
mT_tau = 'sqrt(pow(mytau.Pt()+mymet.Pt(),2)-pow(mytau.Px()+mymet.Px(),2)-pow(mytau.Py()+mymet.Py(),2))'
mT_b1 = 'sqrt(pow(myb1.Pt()+mymet.Pt(),2)-pow(myb1.Px()+mymet.Px(),2)-pow(myb1.Py()+mymet.Py(),2))'
norm_zeta = 'sqrt(pow(mymu.Px()/mymu.Pt()+mytau.Px()/mytau.Pt(),2)+pow(mymu.Py()/mymu.Pt()+mytau.Py()/mytau.Pt(),2))'
x_zeta = '(mymu.Px()/mymu.Pt()+mytau.Px()/mytau.Pt())/norm_zeta'
y_zeta = '(mymu.Py()/mymu.Pt()+mytau.Py()/mytau.Pt())/norm_zeta'
p_zeta_mis = 'mymet.Px()*x_zeta+mymet.Py()*y_zeta'
pzeta_vis = '(mymu.Px()+mytau.Px())*x_zeta+(mymu.Py()+mytau.Py())*y_zeta'
Dzeta = 'p_zeta_mis-0.85*pzeta_vis'


for i in range(len(allfiles)):
    allfiles[i] = allfiles[i].Define('mymu', mymu)\
    .Define('mytau', mytau)\
    .Define('mymet', mymet)\
    .Define('mytt', mytt)\
    .Define('myb1', myb1)\
    .Define('m_btt', m_btt)\
    .Define('m_b1mu', m_b1mu)\
    .Define('m_b1tau', m_b1tau)\
    .Define('dR_tt', dR_tt)\
    .Define('dR_b1mu', dR_b1mu)\
    .Define('dR_b1tau', dR_b1tau)\
    .Define('dR_b1tt', dR_b1tt)\
    .Define('tt_pt', tt_pt)\
    .Define('tt_eta', tt_eta)\
    .Define('mT_mu', mT_mu)\
    .Define('mT_tau', mT_tau)\
    .Define('mT_b1', mT_b1)\
    .Define('norm_zeta', norm_zeta)\
    .Define('x_zeta', x_zeta)\
    .Define('y_zeta', y_zeta)\
    .Define('p_zeta_mis', p_zeta_mis)\
    .Define('pzeta_vis', pzeta_vis)\
    .Define('Dzeta', Dzeta)
tRDF_after = time.perf_counter() 
print(">>> Loaded RDF DataFrame in {} seconds!".format(tRDF_after- tRDF_before))


feature_list = ['pt_1_nominal', 'm_btt', 'm_b1mu', 'm_b1tau',\
                'dR_tt', 'dR_b1mu', 'dR_b1tau', 'dR_b1tt',\
                'tt_pt', 'tt_eta', 'mT_mu', 'mT_tau', 'mT_b1', 'Dzeta',\
                'bpt_deepflavour_1']
    

# H. Compute DNN outputs

#######need to adapt the for loop range when doing tauES and muonES since there are ES values defined only for MC only, embedded only or data only
#######data=range(0,1); embedded=range(1,2); MC=range(2,len(allfiles))

tPre_before = time.perf_counter()
for i in range(len(allfiles)):
    allfiles[i] = pd.DataFrame(allfiles[i].AsNumpy(feature_list))
    allfiles[i] = allfiles[i].values
    allfiles[i] = savedscaler.transform(allfiles[i])
    print(outpaths[i])
tPre_after = time.perf_counter()
print(">>> Formatted RDF dataframe as pandas dataframe in {} seconds...".format(tPre_after - tPre_before))

tEval_before = time.perf_counter()
for i in range(len(allfiles)):
    y_pred = savedmodel.predict(allfiles[i])
    print(type(savedmodel))
    print(type(y_pred))
    y_pred = np.array(y_pred, dtype = [('NN1b', np.float32)]) #######change output names (nominal or which systematics shifted)
    print(type(y_pred))
    file = uproot.recreate(outpaths[i])
    file["mutau_tree_NN"] = y_pred
    print(outpaths[i])
    
tEval_after = time.perf_counter()
print(">>> finished evaluation in {} seconds!".format(tEval_after - tEval_before))

t_after = time.perf_counter()
print(">>> Done in {} seconds".format(t_after - t_before))
