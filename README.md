# Getting started
cmsrel CMSSW_10_2_27
cd CMSSW_10_2_27/src/
cmsenv
git clone https://github.com/llunerti/hnlAnalysis.git
scram b
cd hnlAnalysis/analyzer/

# Generate hnl tree
cmsRun python/ConfFile_cfg_mini_MC.py

