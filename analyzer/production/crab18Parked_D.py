from CRABClient.UserUtilities import config #, getUsernameFromSiteDB
config = config()

#dataset_name = '/ParkingBPH6/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH5/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH4/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH3/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH2/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH1/Run2018A-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH6/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH5/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH4/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH3/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH2/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH1/Run2018B-05May2019-v2/MINIAOD'
#dataset_name = '/ParkingBPH5/Run2018C-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH4/Run2018C-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH3/Run2018C-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH2/Run2018C-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH1/Run2018C-05May2019-v1/MINIAOD'
#dataset_name = '/ParkingBPH5/Run2018D-05May2019promptD-v1/MINIAOD'
#dataset_name = '/ParkingBPH4/Run2018D-05May2019promptD-v1/MINIAOD'
#dataset_name = '/ParkingBPH3/Run2018D-05May2019promptD-v1/MINIAOD'
#dataset_name = '/ParkingBPH2/Run2018D-05May2019promptD-v1/MINIAOD'
dataset_name = '/ParkingBPH1/Run2018D-05May2019promptD-v1/MINIAOD'

config.section_("General")
config.General.workArea = 'crab_projects'
config.General.requestName = str(dataset_name.split("/")[1]+"_"+dataset_name.split("/")[2]+"_v0")

config.section_("JobType")
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = '/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/python/ConfFile_cfg_mini_Run2018D.py'

config.section_("Data")
config.Data.splitting = 'LumiBased'
config.Data.unitsPerJob = 20
config.Data.inputDBS = 'global'
config.Data.inputDataset = dataset_name
config.Data.lumiMask = "/afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt" #https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGoodLumiSectionsJSONFile

config.Data.outputDatasetTag = 'hnlData'

config.section_("Site")
config.Site.storageSite = 'T2_IT_Legnaro'
