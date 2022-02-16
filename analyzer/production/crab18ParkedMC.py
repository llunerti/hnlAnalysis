import json
from CRABClient.UserUtilities import config #, getUsernameFromSiteDB
config = config()

in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau100p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau1000p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau1000p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau100p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = '/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM'
#dataset_name = "/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#dataset_name = "/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v4/MINIAODSIM"
#dataset_name = "/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#dataset_name = "/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
dataset_name = '/ParkingBPH1/Run2018D-05May2019promptD-v1/MINIAOD'
dataset_label = str(dataset_name.split("/")[1])
ver = 1

config.section_("General")
config.General.workArea = 'crab_projects'
config.General.requestName = dataset_label+"_v"+str(ver) 
if str(input_miniAOD_cfg[dataset_name]["dataset_category"]) == "data":
    dataset_label = str(dataset_name.split("/")[1]+"_"+dataset_name.split("/")[2])
    config.General.requestName = dataset_label+"_v"+str(ver)

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/python/hnlAnalyzer_cfg.py'
config.JobType.pyCfgParams = ["globalTag={}".format(str(input_miniAOD_cfg[dataset_name]["global_tag"])),"outputFile=hnlAnalyzer_{}_tree.root".format(dataset_label)]

config.section_("Data")
config.Data.splitting = 'Automatic'
if str(input_miniAOD_cfg[dataset_name]["dataset_category"]) == "data":
   config.Data.lumiMask = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt" #https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
   #config.Data.splitting = 'LumiBased'
   #config.Data.unitsPerJob = 15
   config.Data.splitting = 'EventAwareLumiBased' #each job will contain a varying number of luminosity sections such that the number of events analyzed by each job is roughly unitsPerJob
   config.Data.unitsPerJob = 300000 
config.Data.inputDBS = 'global'
config.Data.inputDataset = dataset_name

config.Data.outputDatasetTag = 'hnl'

config.section_("Site")
config.Site.storageSite = 'T2_IT_Legnaro'
