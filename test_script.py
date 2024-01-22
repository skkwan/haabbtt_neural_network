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
        for i in range(len(self.inpaths)):
            self.allfiles.append(ROOT.RDataFrame(self.path_to_tree, self.inpaths[i]))
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
        #mytt = 'ROOT::Math::PtEtaPhiMVector((mymu+mytau+mymet).Pt(),(mymu+mytau+mymet).Eta(),(mymu+mytau+mymet).Phi(),m_sv)'
        mytt = 'mymu+mytau' #for the moment take only visible parts to reconstruct the ditau system
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
        # dm_a = '(m_bb-m_sv)/m_sv'
        dm_a = '(m_bb-mytt.M())/mytt.M()' #for the moment m_sv = mytt.M()
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
    infolder = 'test_input/'
    outfolder = 'test_output/'

    # inpaths  = ['postprocessed_ntuple_VBFHToTauTau.root']
    inpaths = ['davs://cmsxrootd.hep.wisc.edu:1094/store/user/skkwan/test_svfit_nominal_sync_mutau/test/mt_2018_test-VBFHToTauTau_0.root']
    outpaths = ['dnn_VBFHToTauTau.root']

    for i in range(len(inpaths)):
        # inpaths[i] = infolder + inpaths[i]
        outpaths[i] = outfolder + outpaths[i]


    eval1b_nominal_results = dnnEvaluator("event_tree", "mt1b", "nominal", inpaths, outpaths, "mutau/dnn_score").evaluate()
    eval2b_nominal_results = dnnEvaluator("event_tree", "mt2b", "nominal", inpaths, outpaths, "mutau/dnn_score").evaluate()

    # Write the results into one TTree
    for i in range(len(inpaths)):
        with uproot.recreate(outpaths[i]) as file:
            print(type(eval1b_nominal_results[i]))
            print(eval1b_nominal_results[i])
            file["mutau/dnn_score"] = {"1bNN_nominal": eval1b_nominal_results[i],
                                       "2bNN_nominal": eval2b_nominal_results[i]}


