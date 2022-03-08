# Getting started
cmsrel CMSSW_10_2_27

cd CMSSW_10_2_27/src/

cmsenv

voms-proxy-init --voms cms

git clone https://github.com/llunerti/hnlAnalysis.git

scram b

cd hnlAnalysis/analyzer/

# To generate hnl tree on local
# edit python/run_hnl_tree_production.py with number of events and dataset to process
python python/run_hnl_tree_production.py


# Analyze full dataset via crab
cd production/
# edit crab18ParkedMC.py selecting dataset to process
crab submit -c crab18ParkedMC.py

