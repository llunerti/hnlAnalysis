import json
from CRABClient.UserUtilities import config #, getUsernameFromSiteDB
config = config()

in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

#dataset_name = "/BsToDsNuMu_DsToPhiPi_PhiToMuMu_SoftQCD_TuneCP5_13TeV-pythia8-evtgen/llunerti-RunIISummer20UL18_MiniAOD-dd00e8e5190104a7aafdc4fba9805483/USER"
#dataset_name = "/BsToDsNuMu_DsToNMu_NToMuPi_SoftQCD_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/llunerti-RunIISummer20UL18_MiniAOD-ec01969069464ae5340ffc0d73f20eeb/USER"
dataset_name = "/QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
#dataset_name = "/QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
#dataset_name = "/QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
#dataset_name = "/ParkingBPH6/Run2018B-UL2018_MiniAODv2-v1/MINIAOD"

dataset_label = str(dataset_name.split("/")[1])

sig = int(0)
if input_miniAOD_cfg[dataset_name]["dataset_category"]=="signal":
    sig = 1
ver = 0

config.section_("General")
config.General.workArea = 'crab_projects'
config.General.requestName = dataset_label+"_v"+str(ver) 
if str(input_miniAOD_cfg[dataset_name]["dataset_category"]) == "data":
    dataset_label = str(dataset_name.split("/")[1]+"_"+dataset_name.split("/")[2])
    config.General.requestName = dataset_label+"_v"+str(ver)

config.section_("JobType")
config.JobType.pluginName = 'Analysis'

#cfg_file_full_path = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/python/DsToPhiPi_PhiToMuMu_prompt_cfg.py'
#cfg_file_full_path = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/python/DsToPhiPi_PhiToMuMu_cfg.py'
cfg_file_full_path = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/python/DsToHnlMu_HnlToMuPi_prompt_cfg.py'
#cfg_file_full_path = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/python/hnlAnalyzer_cfg.py'
analyzer_tag = cfg_file_full_path[:cfg_file_full_path.find("_cfg")].split("/")[-1]

config.JobType.psetName = cfg_file_full_path
config.JobType.pyCfgParams = ["globalTag={}".format(str(input_miniAOD_cfg[dataset_name]["global_tag"])),"outputFile={}_{}_tree.root".format(analyzer_tag,dataset_label),"isSignal={}".format(sig)]

config.section_("Data")
config.Data.splitting = 'Automatic'
if dataset_name == "/QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM":
    config.Data.splitting = 'FileBased'
    config.Data.unitsPerJob = 10 
    
if str(input_miniAOD_cfg[dataset_name]["dataset_category"]) == "data":
    config.Data.lumiMask = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt" #https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile
   #config.Data.splitting = 'EventAwareLumiBased' #each job will contain a varying number of luminosity sections such that the number of events analyzed by each job is roughly unitsPerJob
   #config.Data.unitsPerJob = 50000 
config.Data.inputDBS = 'global'
config.Data.inputDataset = dataset_name

config.Data.outputDatasetTag = analyzer_tag

config.section_("Site")
config.Site.storageSite = 'T2_IT_Legnaro'



