import json
import subprocess
from hnlAnalysis.analyzer.tools import *

max_events        = 1000
#das_string        = "/ParkingBPH1/Run2018D-05May2019promptD-v1/MINIAOD"
#das_string        = "/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#das_string        = "/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v4/MINIAODSIM"
#das_string        = "/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
das_string        = "/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#das_string        = "/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM" 
in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"
out_cfg_full_path = "/afs/cern.ch/work/l/llunerti/private/hnlTreeAnalyzer/cfg/hnl_tree_input.json"

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

tot_events = 0
inputFileName_list = []
category = str(input_miniAOD_cfg[das_string]["dataset_category"])

#if max_events=-1 run on first file only
if max_events<0:
    file_name = input_miniAOD_cfg[das_string]["file_name_list"][0].encode('utf-8')
    inputFileName_list = [file_name]
    file_das_dict = json.loads(str(subprocess.check_output('dasgoclient --query='+file_name+' --json', shell=True)))
    tot_events = int(file_das_dict[0]["file"][0]["nevents"])
else:
    tot_events = max_events
    inputFileName_list = [item.encode('utf-8') for item in input_miniAOD_cfg[das_string]["file_name_list"]]

outputFileName = str()

if category == "data":
    outputFileName     = 'hnlAnalyzer_'+das_string.split("/")[1]+"_"+das_string.split("/")[2]+'_tree.root'
else:
    outputFileName     = 'hnlAnalyzer_'+das_string.split("/")[1]+'_tree.root'

# write metadata in a json file
update_json_cfg(das_string,outputFileName,input_miniAOD_cfg,out_cfg_full_path,max_events)

inputFiles_str = ",".join(inputFileName_list[:10]) #a short list of files is sufficient for local tests 

cfgDir = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/python/"
cfgFile = os.path.join(cfgDir,"hnlAnalyzer_cfg.py")

global_tag = input_miniAOD_cfg[das_string]["global_tag"]

command = "cmsRun -e -j report.xml {} inputFiles={} outputFile={} maxEvents={} globalTag={}".format(cfgFile,inputFiles_str,outputFileName,max_events,global_tag)

print("*** Running:")
print(command)

subprocess.call(command,shell=True)
