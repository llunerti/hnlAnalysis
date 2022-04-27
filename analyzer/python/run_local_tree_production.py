import json
import sys
import os
import subprocess
from hnlAnalysis.analyzer.tools import *

cmsrun_cfg_name = sys.argv[1]
cfg_label = cmsrun_cfg_name[:cmsrun_cfg_name.find("_cfg.py")].split("/")[-1]

max_events        = 10
#das_string        = "/BsToDsNuMu_DsToPhiPi_PhiToMuMu_SoftQCD_TuneCP5_13TeV-pythia8-evtgen/llunerti-RunIISummer20UL18_MiniAOD-dd00e8e5190104a7aafdc4fba9805483/USER"
#das_string        = "/BsToDsNuMu_DsToNMu_NToMuPi_SoftQCD_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/llunerti-RunIISummer20UL18_MiniAOD-ec01969069464ae5340ffc0d73f20eeb/USER"
#das_string        = "/QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
#das_string        = "/QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
#das_string        = "/QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM"
das_string        = "/ParkingBPH6/Run2018B-UL2018_MiniAODv2-v1/MINIAOD"
in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"
out_cfg_full_path = "/afs/cern.ch/work/l/llunerti/private/hnlTreeAnalyzer/cfg/" + cfg_label + "_UL_tree_input.json"

if not os.path.isfile(out_cfg_full_path):
    subprocess.call("echo '{{}}' >> {}".format(out_cfg_full_path),shell=True)

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

tot_events = 0
inputFileName_list = []
category = str(input_miniAOD_cfg[das_string]["dataset_category"])

#if max_events=-1 run on first file only
if max_events<0:
    inputFileName_list = [item.encode('utf-8') for item in input_miniAOD_cfg[das_string]["file_name_list"][:1]]
    for file_name in inputFileName_list:
        file_das_dict = json.loads(str(subprocess.check_output('dasgoclient --query='+file_name+' --json', shell=True)))
        tot_events += int(file_das_dict[0]["file"][0]["nevents"])
else:
    tot_events = max_events
    inputFileName_list = [item.encode('utf-8') for item in input_miniAOD_cfg[das_string]["file_name_list"]]

outputFileName = str()

if category == "data":
    outputFileName = cfg_label+'_'+das_string.split("/")[1]+"_"+das_string.split("/")[2]+'_tree.root'
else:
    outputFileName = cfg_label+'_'+das_string.split("/")[1]+'_tree.root'

output_dir = subprocess.check_output("pwd",shell=True).strip("\n")
outputFileNameList = [str(os.path.join(output_dir,outputFileName))]

# write metadata in a json file
update_json_cfg(das_string,outputFileNameList,input_miniAOD_cfg,out_cfg_full_path,max_events)

inputFiles_str = ",".join(inputFileName_list[:100]) #a short list of files is sufficient for local tests 

cfgFile = os.path.join(str(os.getcwd()),cmsrun_cfg_name)

global_tag = input_miniAOD_cfg[das_string]["global_tag"]

sig = int(0)
if input_miniAOD_cfg[das_string]["dataset_category"]=="signal":
    sig = 1

command = "cmsRun -e -j report.xml {} inputFiles={} outputFile={} maxEvents={} globalTag={} isSignal={}".format(cfgFile,inputFiles_str,outputFileName,max_events,global_tag,sig)

print("*** Running:")
print("TOTAL EVENTS TO RUN: {}".format(tot_events))
print(command)

subprocess.call(command,shell=True)
