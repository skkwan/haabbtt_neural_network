# Sandbox for trying to add a numpy array as a branch to RDataFrame in Python.
#
# Usage: python sandbox.py


import ROOT

inpathsMuTau = ['/eos/user/s/skkwan/hToAA/svfitted/SUSYVBFHToAA_AToBB_AToTauTau_M-45/all_2018_SUSYVBFHToAA_AToBB_AToTauTau_M-45-postprocessed_ntuple_SUSYVBFHToAA_AToBB_AToTauTau_M-45_0.root']


# Write the results into one TTree
for i in range(len(inpathsMuTau)):

    # Get the input file as RDataFrame
    thisDf = ROOT.RDataFrame("mutau/event_tree", inpathsMuTau[i])     

    # Fake scores to add as a branch
    dummyScores = np.zeros(thisDf.Count())   

    # How to add dummyScores as a branch?

    # This is how we can write out the RDataFrame to a new .ROOT file
    thisDf.Snapshot("mutau/event_tree", "new.root")

