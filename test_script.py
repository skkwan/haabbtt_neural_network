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

# def getBranch(treeName, targetBranchName, allFiles):
#     """
#     Get a specific branch from a tree.
#     """
#     returnValue = []
#     for filename in allFiles:
#         f = ROOT.TFile(filename)
#         t = f.Get(treeName)
#         if t:
            

def preselectFilesWithTree(allFiles, treeName):
    """
    Given a list of paths to files (allFiles), return the subset of them (as a list of strings) that contain a valid specified tree.
    """
    goodFiles = []
    for filename in allFiles:
        f = ROOT.TFile(filename)
        t = f.Get(treeName)
        if t:
            goodFiles.append(filename)
    return goodFiles

def resetBit(filename, treeName): 
    """
    Reset kEntriesReshuffled of a tree, given a list of paths to the ROOT files.
    """
    print("resetBit: doing {} {} ".format(filename, treeName))
    f = ROOT.TFile(filename)
    t = f.Get(treeName)
    if not t:
        print("resetBit: file {} does not contain TTree {}, continue".format(filename, treeName))
    else:
        print(t.TestBit(ROOT.TTree.EStatusBits.kEntriesReshuffled))
        t.ResetBit(ROOT.TTree.EStatusBits.kEntriesReshuffled)
        print(t.TestBit(ROOT.TTree.EStatusBits.kEntriesReshuffled))



class dnnEvaluator:
    """
    Definition of a class for evaluating a trained DNN.
    path_to_tree: path to tree inside the input file.
    model_name: mt1b or mt2b, determines the path to the scaler/model to load.
    """
    def __init__(self, path_to_tree, model_name, sys_name, inpaths, outpaths, output_tree_name):
        t0 = time.perf_counter()
        self.path_to_tree = path_to_tree
        self.model_name = model_name
        self.sys_name = sys_name
        self.inpaths = inpaths
        self.outpaths = outpaths
        self.output_tree_name = output_tree_name
        self.savedscaler = joblib.load('trained_models_rerunBtag/{}_scaler.gz'.format(self.model_name))
        self.savedmodel = keras.models.load_model('trained_models_rerunBtag/{}_model'.format(self.model_name))
        self.savedmodel.summary()
        t1 = time.perf_counter()
        print(">>> dnnEvaluator initializer: Loaded model for {} in {} seconds!".format(self.model_name, t1 - t0))
        # Load input files as RDF
        self.allfiles = []
        self.arrayRunLumiEvent = []

        for i in range(len(self.inpaths)):
            thisDf = ROOT.RDataFrame(self.path_to_tree, self.inpaths[i])
            self.allfiles.append(thisDf)
            self.arrayRunLumiEvent.append(thisDf.AsNumpy(["run", "lumi", "evt"]))
        t2 = time.perf_counter()
        print(">>> dnnEvaluator initializer: Loaded Tree {} in files as a list of RDF in {} seconds!".format(self.path_to_tree, t2 - t1))


    def evaluate_model(self):
        """
        Helper function to perform the evaluation. Called by evaluate_mt_NN1b. Returns the results.
        Description is e.g. "nominal".
        """
        t0 = time.perf_counter()
        for i in range(len(self.allfiles)):
            self.allfiles[i] = pd.DataFrame(self.allfiles[i].AsNumpy(self.feature_list))
            self.allfiles[i] = self.allfiles[i].values
            self.allfiles[i] = self.savedscaler.transform(self.allfiles[i])
            print(self.outpaths[i])
        t1 = time.perf_counter()
        results = []
        print(">>> Formatted RDF dataframe as pandas dataframe for {} {} in {} seconds...".format(self.model_name, self.sys_name, t1 - t0))
        for i in range(len(self.allfiles)):
            y_pred = self.savedmodel.predict(self.allfiles[i])
            y_pred = np.array(y_pred, dtype = [("score", np.float32)]) #######change output names (nominal or which systematics shifted)
            results.append(y_pred)       
        t2 = time.perf_counter()
        print(">>> evaluate(): Finished evaluation of {} {} in {} seconds!".format(self.model_name, self.sys_name, t2 - t1))
        return results

    def evaluate_mt_NN1b(self):
        """
        Perform the evaluation for mutau NN1b. No return value.
        """
        t0 = time.perf_counter() 
        # Nominal variables for 1bNN
        mymu = 'ROOT::Math::PtEtaPhiMVector(pt_1_nominal, eta_1, phi_1, m_1_nominal)'
        mytau = 'ROOT::Math::PtEtaPhiMVector(pt_2_nominal, eta_2, phi_2, m_2_nominal)'
        mymet = 'ROOT::Math::PtEtaPhiMVector(met_nominal, 0, metphi_nominal, 0)'
        # Use m_sv
        mytt = 'ROOT::Math::PtEtaPhiMVector((mymu+mytau+mymet).Pt(),(mymu+mytau+mymet).Eta(),(mymu+mytau+mymet).Phi(),m_sv)'
        # mytt = 'mymu+mytau'#for the moment take only visible parts to reconstruct the ditau system
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

        for i in range(len(self.allfiles)):
            self.allfiles[i] = self.allfiles[i]\
            .Filter("channel == 0", ">>> evaluate_mt_NN1b: select mutau channel")\
            .Define('mymu', mymu)\
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
        self.feature_list = ['pt_1_nominal', 'm_btt', 'm_b1mu', 'm_b1tau',\
                             'dR_tt', 'dR_b1mu', 'dR_b1tau', 'dR_b1tt',\
                             'tt_pt', 'tt_eta', 'mT_mu', 'mT_tau', 'mT_b1', 'Dzeta',\
                             'bpt_deepflavour_1']
        results = self.evaluate_model()
        t1 = time.perf_counter() 
        print(">>> Evaluated mutau NN1b in {} seconds!".format(t1 - t0))
        return results

    def evaluate_mt_NN2b(self):
        """
        Perform the evaluation for mutau NN2b.
        """
        t_before = time.perf_counter()
        mymu = 'ROOT::Math::PtEtaPhiMVector(pt_1_nominal, eta_1, phi_1, m_1_nominal)'
        mytau = 'ROOT::Math::PtEtaPhiMVector(pt_2_nominal, eta_2, phi_2, m_2_nominal)'
        mymet = 'ROOT::Math::PtEtaPhiMVector(met_nominal, 0, metphi_nominal, 0)'
        # Use m_sv
        mytt = 'ROOT::Math::PtEtaPhiMVector((mymu+mytau+mymet).Pt(),(mymu+mytau+mymet).Eta(),(mymu+mytau+mymet).Phi(),m_sv)'
        # mytt = 'mymu+mytau' #for the moment take only visible parts to reconstruct the ditau system
        myb1 = 'ROOT::Math::PtEtaPhiMVector(bpt_deepflavour_1,beta_deepflavour_1,bphi_deepflavour_1,bm_deepflavour_1)'
        myb2 = 'ROOT::Math::PtEtaPhiMVector(bpt_deepflavour_2,beta_deepflavour_2,bphi_deepflavour_2,bm_deepflavour_2)'
        m_b1tt = '(mytt+myb1).M()'
        m_b2tt = '(mytt+myb2).M()'
        m_bbtt = '(mytt+myb1+myb2).M()'
        m_bb = '(myb1+myb2).M()'
        m_b1mu = '(mymu+myb1).M()'
        m_b1tau = '(mytau+myb1).M()'
        m_b2mu = '(mymu+myb2).M()'
        m_b2tau = '(mytau+myb2).M()'
        m_bbmu = '(myb1+myb2+mymu).M()'
        m_bbtau = '(myb1+myb2+mytau).M()'
        # Use m_sv
        dm_a = '(m_bb-m_sv)/m_sv'
        # dm_a = '(m_bb-mytt.M())/mytt.M()' #for the moment m_sv = mytt.M()
        dR_tt = 'ROOT::Math::VectorUtil::DeltaR(mymu,mytau)'
        dR_b1mu = 'ROOT::Math::VectorUtil::DeltaR(mymu,myb1)'
        dR_b1tau = 'ROOT::Math::VectorUtil::DeltaR(mytau,myb1)'
        dR_b2mu = 'ROOT::Math::VectorUtil::DeltaR(mymu,myb2)'
        dR_b2tau = 'ROOT::Math::VectorUtil::DeltaR(mytau,myb2)'
        dR_bb = 'ROOT::Math::VectorUtil::DeltaR(myb1,myb2)'
        dR_b1tt = 'ROOT::Math::VectorUtil::DeltaR(myb1,mytt)'
        dR_b2tt = 'ROOT::Math::VectorUtil::DeltaR(myb2,mytt)'
        dR_bbmu = 'ROOT::Math::VectorUtil::DeltaR(myb1+myb2,mymu)'
        dR_bbtau = 'ROOT::Math::VectorUtil::DeltaR(myb1+myb2,mytau)'
        dR_aa = 'ROOT::Math::VectorUtil::DeltaR(mytt,myb1+myb2)'

        mT_mu = 'sqrt(pow(mymu.Pt()+mymet.Pt(),2)-pow(mymu.Px()+mymet.Px(),2)-pow(mymu.Py()+mymet.Py(),2))'
        mT_tau = 'sqrt(pow(mytau.Pt()+mymet.Pt(),2)-pow(mytau.Px()+mymet.Px(),2)-pow(mytau.Py()+mymet.Py(),2))'
        mT_b1 = 'sqrt(pow(myb1.Pt()+mymet.Pt(),2)-pow(myb1.Px()+mymet.Px(),2)-pow(myb1.Py()+mymet.Py(),2))'
        mT_b2 = 'sqrt(pow(myb2.Pt()+mymet.Pt(),2)-pow(myb2.Px()+mymet.Px(),2)-pow(myb2.Py()+mymet.Py(),2))'
        norm_zeta = 'sqrt(pow(mymu.Px()/mymu.Pt()+mytau.Px()/mytau.Pt(),2)+pow(mymu.Py()/mymu.Pt()+mytau.Py()/mytau.Pt(),2))'
        x_zeta = '(mymu.Px()/mymu.Pt()+mytau.Px()/mytau.Pt())/norm_zeta'
        y_zeta = '(mymu.Py()/mymu.Pt()+mytau.Py()/mytau.Pt())/norm_zeta'
        p_zeta_mis = 'mymet.Px()*x_zeta+mymet.Py()*y_zeta'
        pzeta_vis = '(mymu.Px()+mytau.Px())*x_zeta+(mymu.Py()+mytau.Py())*y_zeta'
        Dzeta = 'p_zeta_mis-0.85*pzeta_vis'


        for i in range(len(self.allfiles)):
            self.allfiles[i] = self.allfiles[i]\
            .Filter("channel == 0", ">>> evaluate_mt_NN2b: select mutau channel")\
            .Define('mymu', mymu)\
            .Define('mytau', mytau)\
            .Define('mymet', mymet)\
            .Define('mytt', mytt)\
            .Define('myb1', myb1)\
            .Define('myb2', myb2)\
            .Define('m_b1tt', m_b1tt)\
            .Define('m_b2tt', m_b2tt)\
            .Define('m_bbtt', m_bbtt)\
            .Define('m_bb', m_bb)\
            .Define('m_b1mu', m_b1mu)\
            .Define('m_b1tau', m_b1tau)\
            .Define('m_b2mu', m_b2mu)\
            .Define('m_b2tau', m_b2tau)\
            .Define('m_bbmu', m_bbmu)\
            .Define('m_bbtau', m_bbtau)\
            .Define('dm_a', dm_a)\
            .Define('dR_tt', dR_tt)\
            .Define('dR_b1mu', dR_b1mu)\
            .Define('dR_b1tau', dR_b1tau)\
            .Define('dR_b2mu', dR_b2mu)\
            .Define('dR_b2tau', dR_b2tau)\
            .Define('dR_bb', dR_bb)\
            .Define('dR_b1tt', dR_b1tt)\
            .Define('dR_b2tt', dR_b2tt)\
            .Define('dR_aa', dR_aa)\
            .Define('dR_bbmu', dR_bbmu)\
            .Define('dR_bbtau', dR_bbtau)\
            .Define('mT_mu', mT_mu)\
            .Define('mT_tau', mT_tau)\
            .Define('mT_b1', mT_b1)\
            .Define('mT_b2', mT_b2)\
            .Define('norm_zeta', norm_zeta)\
            .Define('x_zeta', x_zeta)\
            .Define('y_zeta', y_zeta)\
            .Define('p_zeta_mis', p_zeta_mis)\
            .Define('pzeta_vis', pzeta_vis)\
            .Define('Dzeta', Dzeta)

        self.feature_list = ['pt_1_nominal', 'm_b1tt', 'm_b2tt', 'm_bbtt', 'm_bb', 'm_b1mu', 'm_b1tau', 'm_b2mu', 'm_b2tau', 'dm_a',\
                            'm_bbmu', 'm_bbtau', 'dR_tt', 'dR_b1mu', 'dR_b1tau', 'dR_b2mu', 'dR_b2tau', 'dR_bb', 'dR_b1tt', 'dR_b2tt', 'dR_aa',\
                            'dR_bbmu','dR_bbtau', 'mT_mu', 'mT_tau', 'mT_b1', 'mT_b2', 'Dzeta',\
                            'bpt_deepflavour_1', 'bpt_deepflavour_2']
        results = self.evaluate_model()
        t_after = time.perf_counter() 
        print(">>> Evaluated mutau NN2b in {} seconds!".format(t_after - t_before))
        return results

    def evaluate(self):
        if (self.model_name == "mt1b"):
            return self.evaluate_mt_NN1b()
        elif (self.model_name == "mt2b"):
            return self.evaluate_mt_NN2b()
    
    def writeResults(self):
        """
        Write the results.
        """
        # self.results has a dictionary of ["nominal"] which maps to a list of results
        with uproot.recreate(self.outpaths[i]) as file:
            file[self.output_tree_name] = {branchname: y_pred}
      

if __name__ == "__main__":
    # Get the input files 
    infolder = '/eos/user/s/skkwan/hToAA/svfitted/'
    outfolder = '/eos/user/s/skkwan/hToAA/svfitted/'

    # inpaths  = ['postprocessed_ntuple_VBFHToTauTau.root']
    inpathsToTry = [
    # 'DY1JetsToLL/all_2018_DY1JetsToLL-postprocessed_ntuple_DY1JetsToLL_0.root',
    # 'DY2JetsToLL/all_2018_DY2JetsToLL-postprocessed_ntuple_DY2JetsToLL_0.root',
    # 'DY3JetsToLL/all_2018_DY3JetsToLL-postprocessed_ntuple_DY3JetsToLL_0.root',
    # 'DY4JetsToLL/all_2018_DY4JetsToLL-postprocessed_ntuple_DY4JetsToLL_0.root',
    # 'DYJetsToLL_M-10to50/all_2018_DYJetsToLL_M-10to50-postprocessed_ntuple_DYJetsToLL_M-10to50_0.root',
    # 'DYJetsToLL_M-50/all_2018_DYJetsToLL_M-50-postprocessed_ntuple_DYJetsToLL_M-50_0.root',
    # 'EGamma-Run2018A/all_2018_EGamma-Run2018A-postprocessed_ntuple_EGamma-Run2018A_0.root',
    # 'EGamma-Run2018A/all_2018_EGamma-Run2018A-postprocessed_ntuple_EGamma-Run2018A_1.root',
    # 'EGamma-Run2018B/all_2018_EGamma-Run2018B-postprocessed_ntuple_EGamma-Run2018B_0.root',
    # 'EGamma-Run2018B/all_2018_EGamma-Run2018B-postprocessed_ntuple_EGamma-Run2018B_1.root',
    # 'EGamma-Run2018B/all_2018_EGamma-Run2018B-postprocessed_ntuple_EGamma-Run2018B_2.root',
    # 'EGamma-Run2018C/all_2018_EGamma-Run2018C-postprocessed_ntuple_EGamma-Run2018C_0.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_0.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_1.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_2.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_3.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_4.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_5.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_6.root',
    # 'EGamma-Run2018D/all_2018_EGamma-Run2018D-postprocessed_ntuple_EGamma-Run2018D_7.root',
    # 'Embedded-Run2018A-EMu/all_2018_Embedded-Run2018A-EMu-postprocessed_ntuple_Embedded-Run2018A-EMu_0.root',
    # 'Embedded-Run2018A-EMu/all_2018_Embedded-Run2018A-EMu-postprocessed_ntuple_Embedded-Run2018A-EMu_1.root',
    # 'Embedded-Run2018A-EMu/all_2018_Embedded-Run2018A-EMu-postprocessed_ntuple_Embedded-Run2018A-EMu_2.root',
    # 'Embedded-Run2018A-EMu/all_2018_Embedded-Run2018A-EMu-postprocessed_ntuple_Embedded-Run2018A-EMu_3.root',
    # 'Embedded-Run2018A-ElTau/all_2018_Embedded-Run2018A-ElTau-postprocessed_ntuple_Embedded-Run2018A-ElTau_0.root',
    # 'Embedded-Run2018A-MuTau/all_2018_Embedded-Run2018A-MuTau-postprocessed_ntuple_Embedded-Run2018A-MuTau_0.root',
    # 'Embedded-Run2018B-EMu/all_2018_Embedded-Run2018B-EMu-postprocessed_ntuple_Embedded-Run2018B-EMu_0.root',
    # 'Embedded-Run2018B-ElTau/all_2018_Embedded-Run2018B-ElTau-postprocessed_ntuple_Embedded-Run2018B-ElTau_0.root',
    # 'Embedded-Run2018B-MuTau/all_2018_Embedded-Run2018B-MuTau-postprocessed_ntuple_Embedded-Run2018B-MuTau_0.root',
    # 'Embedded-Run2018C-EMu/all_2018_Embedded-Run2018C-EMu-postprocessed_ntuple_Embedded-Run2018C-EMu_0.root',
    # 'Embedded-Run2018C-ElTau/all_2018_Embedded-Run2018C-ElTau-postprocessed_ntuple_Embedded-Run2018C-ElTau_0.root',
    # 'Embedded-Run2018C-MuTau/all_2018_Embedded-Run2018C-MuTau-postprocessed_ntuple_Embedded-Run2018C-MuTau_0.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_0.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_1.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_2.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_3.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_4.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_5.root',
    # 'Embedded-Run2018D-EMu/all_2018_Embedded-Run2018D-EMu-postprocessed_ntuple_Embedded-Run2018D-EMu_6.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_0.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_1.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_2.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_3.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_4.root',
    # 'Embedded-Run2018D-ElTau/all_2018_Embedded-Run2018D-ElTau-postprocessed_ntuple_Embedded-Run2018D-ElTau_5.root',
    # 'Embedded-Run2018D-MuTau/all_2018_Embedded-Run2018D-MuTau-postprocessed_ntuple_Embedded-Run2018D-MuTau_0.root',
    # 'Embedded-Run2018D-MuTau/all_2018_Embedded-Run2018D-MuTau-postprocessed_ntuple_Embedded-Run2018D-MuTau_1.root',
    # 'Embedded-Run2018D-MuTau/all_2018_Embedded-Run2018D-MuTau-postprocessed_ntuple_Embedded-Run2018D-MuTau_2.root',
    # 'Embedded-Run2018D-MuTau/all_2018_Embedded-Run2018D-MuTau-postprocessed_ntuple_Embedded-Run2018D-MuTau_3.root',
    # 'Embedded-Run2018D-MuTau/all_2018_Embedded-Run2018D-MuTau-postprocessed_ntuple_Embedded-Run2018D-MuTau_4.root',
    # 'GluGluHToTauTau/all_2018_GluGluHToTauTau-postprocessed_ntuple_GluGluHToTauTau_0.root',
    # 'GluGluHToWWTo2L2Nu/all_2018_GluGluHToWWTo2L2Nu-postprocessed_ntuple_GluGluHToWWTo2L2Nu_0.root',
    # 'GluGluZH_HToWWTo2L2Nu/all_2018_GluGluZH_HToWWTo2L2Nu-postprocessed_ntuple_GluGluZH_HToWWTo2L2Nu_0.root',
    # 'GluGluZH_HToWW_ZTo2L/all_2018_GluGluZH_HToWW_ZTo2L-postprocessed_ntuple_GluGluZH_HToWW_ZTo2L_0.root',
    # 'HWminusJ_HToWW/all_2018_HWminusJ_HToWW-postprocessed_ntuple_HWminusJ_HToWW_0.root',
    # 'HWplusJ_HToWW/all_2018_HWplusJ_HToWW-postprocessed_ntuple_HWplusJ_HToWW_0.root',
    # 'HZJ_HToWW/all_2018_HZJ_HToWW-postprocessed_ntuple_HZJ_HToWW_0.root',
    # 'MuonEG-Run2018A/all_2018_MuonEG-Run2018A-postprocessed_ntuple_MuonEG-Run2018A_0.root',
    # 'MuonEG-Run2018B/all_2018_MuonEG-Run2018B-postprocessed_ntuple_MuonEG-Run2018B_0.root',
    # 'MuonEG-Run2018C/all_2018_MuonEG-Run2018C-postprocessed_ntuple_MuonEG-Run2018C_0.root',
    # 'MuonEG-Run2018D/all_2018_MuonEG-Run2018D-postprocessed_ntuple_MuonEG-Run2018D_0.root',
    # 'ST_t-channel_antitop/all_2018_ST_t-channel_antitop-postprocessed_ntuple_ST_t-channel_antitop_0.root',
    # 'ST_t-channel_top/all_2018_ST_t-channel_top-postprocessed_ntuple_ST_t-channel_top_0.root',
    # 'ST_t-channel_top/all_2018_ST_t-channel_top-postprocessed_ntuple_ST_t-channel_top_1.root',
    # 'ST_t-channel_top/all_2018_ST_t-channel_top-postprocessed_ntuple_ST_t-channel_top_2.root',
    # 'ST_tW_antitop/all_2018_ST_tW_antitop-postprocessed_ntuple_ST_tW_antitop_0.root',
    # 'ST_tW_top/all_2018_ST_tW_top-postprocessed_ntuple_ST_tW_top_0.root',
    # 'SUSYGluGluToHToAA_AToBB_AToTauTau_M-45/all_2018_SUSYGluGluToHToAA_AToBB_AToTauTau_M-45-postprocessed_ntuple_SUSYGluGluToHToAA_AToBB_AToTauTau_M-45_0.root',
    'SUSYVBFHToAA_AToBB_AToTauTau_M-45/all_2018_SUSYVBFHToAA_AToBB_AToTauTau_M-45-postprocessed_ntuple_SUSYVBFHToAA_AToBB_AToTauTau_M-45_0.root',
    # 'SingleMuon-Run2018A/all_2018_SingleMuon-Run2018A-postprocessed_ntuple_SingleMuon-Run2018A_0.root',
    # 'SingleMuon-Run2018A/all_2018_SingleMuon-Run2018A-postprocessed_ntuple_SingleMuon-Run2018A_1.root',
    # 'SingleMuon-Run2018B/all_2018_SingleMuon-Run2018B-postprocessed_ntuple_SingleMuon-Run2018B_0.root',
    # 'SingleMuon-Run2018B/all_2018_SingleMuon-Run2018B-postprocessed_ntuple_SingleMuon-Run2018B_1.root',
    # 'SingleMuon-Run2018C/all_2018_SingleMuon-Run2018C-postprocessed_ntuple_SingleMuon-Run2018C_0.root',
    # 'SingleMuon-Run2018D/all_2018_SingleMuon-Run2018D-postprocessed_ntuple_SingleMuon-Run2018D_0.root',
    # 'SingleMuon-Run2018D/all_2018_SingleMuon-Run2018D-postprocessed_ntuple_SingleMuon-Run2018D_1.root',
    # 'SingleMuon-Run2018D/all_2018_SingleMuon-Run2018D-postprocessed_ntuple_SingleMuon-Run2018D_2.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_0.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_1.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_10.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_11.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_12.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_13.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_14.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_15.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_16.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_17.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_18.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_19.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_2.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_20.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_21.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_22.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_23.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_24.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_25.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_26.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_27.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_28.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_29.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_3.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_30.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_4.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_5.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_6.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_7.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_8.root',
    # 'TTTo2L2Nu/all_2018_TTTo2L2Nu-postprocessed_ntuple_TTTo2L2Nu_9.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_0.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_1.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_2.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_3.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_4.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_5.root',
    # 'TTToHadronic/all_2018_TTToHadronic-postprocessed_ntuple_TTToHadronic_6.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_0.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_1.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_10.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_11.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_12.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_13.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_14.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_15.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_16.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_17.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_18.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_19.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_2.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_20.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_21.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_22.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_23.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_24.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_25.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_26.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_27.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_28.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_29.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_3.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_30.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_31.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_32.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_33.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_34.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_35.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_36.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_37.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_38.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_39.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_4.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_40.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_41.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_42.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_43.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_44.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_45.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_46.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_47.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_48.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_49.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_5.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_50.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_51.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_52.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_53.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_54.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_55.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_56.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_57.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_58.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_59.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_6.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_60.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_61.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_62.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_63.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_64.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_65.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_66.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_67.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_68.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_69.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_7.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_70.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_71.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_72.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_73.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_74.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_75.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_76.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_77.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_78.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_8.root',
    # 'TTToSemiLeptonic/all_2018_TTToSemiLeptonic-postprocessed_ntuple_TTToSemiLeptonic_9.root',
    # 'VBFHToTauTau/all_2018_VBFHToTauTau-postprocessed_ntuple_VBFHToTauTau_0.root',
    # 'VBFHToWWTo2L2Nu/all_2018_VBFHToWWTo2L2Nu-postprocessed_ntuple_VBFHToWWTo2L2Nu_0.root',
    # 'VVTo2L2Nu/all_2018_VVTo2L2Nu-postprocessed_ntuple_VVTo2L2Nu_0.root',
    # 'VVTo2L2Nu/all_2018_VVTo2L2Nu-postprocessed_ntuple_VVTo2L2Nu_1.root',
    # 'VVTo2L2Nu/all_2018_VVTo2L2Nu-postprocessed_ntuple_VVTo2L2Nu_2.root',
    # 'W1JetsToLNu/all_2018_W1JetsToLNu-postprocessed_ntuple_W1JetsToLNu_0.root',
    # 'W2JetsToLNu/all_2018_W2JetsToLNu-postprocessed_ntuple_W2JetsToLNu_0.root',
    # 'W3JetsToLNu/all_2018_W3JetsToLNu-postprocessed_ntuple_W3JetsToLNu_0.root',
    # 'W4JetsToLNu/all_2018_W4JetsToLNu-postprocessed_ntuple_W4JetsToLNu_0.root',
    # 'WJetsToLNu/all_2018_WJetsToLNu-postprocessed_ntuple_WJetsToLNu_0.root',
    # 'WZTo2Q2L/all_2018_WZTo2Q2L-postprocessed_ntuple_WZTo2Q2L_0.root',
    # 'WZTo3LNu/all_2018_WZTo3LNu-postprocessed_ntuple_WZTo3LNu_0.root',
    # 'WminusHToTauTau/all_2018_WminusHToTauTau-postprocessed_ntuple_WminusHToTauTau_0.root',
    # 'WplusHToTauTau/all_2018_WplusHToTauTau-postprocessed_ntuple_WplusHToTauTau_0.root',
    # 'ZHToTauTau/all_2018_ZHToTauTau-postprocessed_ntuple_ZHToTauTau_0.root',
    # 'ZZTo2Q2L/all_2018_ZZTo2Q2L-postprocessed_ntuple_ZZTo2Q2L_0.root',
    # 'ZZTo4L/all_2018_ZZTo4L-postprocessed_ntuple_ZZTo4L_0.root',
    # 'ttHToNonbb/all_2018_ttHToNonbb-postprocessed_ntuple_ttHToNonbb_0.root',
    # 'ttHTobb/all_2018_ttHTobb-postprocessed_ntuple_ttHTobb_0.root',
    # 'ttHTobb/all_2018_ttHTobb-postprocessed_ntuple_ttHTobb_1.root',
    # 'ttHTobb/all_2018_ttHTobb-postprocessed_ntuple_ttHTobb_2.root',
    # 'ttHTobb/all_2018_ttHTobb-postprocessed_ntuple_ttHTobb_3.root'
    ]

    # Initialize
    for i in range(len(inpathsToTry)):
        inpathsToTry[i] = infolder + inpathsToTry[i]

    # Make list of files that have mutau/event_tree
    inpathsMuTau = preselectFilesWithTree(inpathsToTry, "mutau/event_tree")

    outpathsMuTau = [""] * len(inpathsMuTau)

    # print(inpathsMuTau)
    print(type(inpathsMuTau[0]))

    for i in range(len(inpathsMuTau)):
        # Reset "shuffled" bit
        resetBit(inpathsMuTau[i], "mutau/event_tree")
        # sampleName = inpaths[i].split("/")[0]
       # outpaths[i] = inpaths[i][:]
        outpathsMuTau[i] = inpathsMuTau[i][:].replace("all_2018","dnn_score").replace("postprocessed_ntuple", "dnn_score")
        print(">>> Input path: {}, output path: {}".format(inpathsMuTau[i], outpathsMuTau[i]))

    my1b = dnnEvaluator("mutau/event_tree", "mt1b", "nominal", inpathsMuTau, outpathsMuTau, "mutau/dnn_score")
    my2b = dnnEvaluator("mutau/event_tree", "mt2b", "nominal", inpathsMuTau, outpathsMuTau, "mutau/dnn_score")
    eval1b_nominal_results = my1b.evaluate()
    eval2b_nominal_results = my2b.evaluate()

    # Write the results into one TTree
    for i in range(len(inpathsMuTau)):
        with uproot.recreate(outpathsMuTau[i]) as file:
            # print(type(eval1b_nominal_results[i]))
            # print(eval1b_nominal_results[i])
            file["mutau/dnn_score"] = {"1bNN_nominal": eval1b_nominal_results[i],
                                       "2bNN_nominal": eval2b_nominal_results[i],
                                       "run":  my1b.arrayRunLumiEvent[i]["run"],
                                       "lumi": my1b.arrayRunLumiEvent[i]["lumi"],
                                       "evt":  my1b.arrayRunLumiEvent[i]["evt"]}


