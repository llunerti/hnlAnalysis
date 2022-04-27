import subprocess
import json
import os
import sys
from hnlAnalysis.analyzer.tools import *

crab_dir = sys.argv[1]
crab_log = os.path.join(crab_dir,"crab.log")

s = subprocess.check_output('cat {} | grep "^config.JobType.psetName"'.format(crab_log),shell=True)
cmsrun_cfg_name = s.split(" ")[-1]
cfg_label = cmsrun_cfg_name[:cmsrun_cfg_name.find("_cfg.py")].split("/")[-1]



in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_6_30/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"
out_cfg_full_path = "/afs/cern.ch/work/l/llunerti/private/hnlTreeAnalyzer/cfg/"+ cfg_label +"_UL_tree_input_fromCrab.json"

if not os.path.isfile(out_cfg_full_path):
    subprocess.call("echo '{{}}' >> {}".format(out_cfg_full_path),shell=True)

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

##get number of jobs from crab status output saved into crab log
##this command is luckly being run every new submission (couldn't find another way)
#n_jobs_str = subprocess.check_output('cat {} | grep "([0-9]*/[0-9]*)" -o | head -n 1'.format(crab_log),shell=True)
#if n_jobs_str == "":
#    print("Couldn't find number of jobs in {}, please check if crab status has been run".format(crab_log))
#    exit(1)
#
#n_jobs = int(n_jobs_str.split("/")[-1].strip("\n").strip(")"))
#
#file_list = []
#make_list_of_lists = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)] # https://stackoverflow.com/questions/4119070/how-to-divide-a-list-into-n-equal-parts-python
#
## crab getoutput can drop a max of 500 files at the time
#if n_jobs>500:
#    job_range = range(1,n_jobs+1)
#    list_of_jobids = make_list_of_lists(job_range,500)
#    for jobids in list_of_jobids:
#        jobid_range = str(jobids[0])+"-"+str(jobids[-1])
#        print("Processing {} job ids...".format(jobid_range))
#        fl = subprocess.check_output('crab getoutput -d {} --jobids={} --dump | grep "PFN"'.format(crab_dir,jobid_range),shell=True)
#        fl = fl.split("\n")[0:-1]
#        fl = [x.split(" ")[-1] for x in fl]
#        file_list += fl
#        
#else:
#    file_list = subprocess.check_output('crab getoutput -d {} --dump | grep "PFN"'.format(crab_dir),shell=True)
#    
#    # manipulate output in order to get a list of PFN only
#    file_list = file_list.split("\n")[0:-1]
#    file_list = [x.split(" ")[-1] for x in file_list]

file_list = subprocess.check_output('crab getoutput -d {} --jobids=1 --dump | grep "PFN"'.format(crab_dir),shell=True)

# manipulate output in order to get a list of PFN only
file_list = file_list.split("\n")[0:-1]
file_list = [x.split(" ")[-1] for x in file_list]

n_str = subprocess.check_output('cat {} | grep "Number of events read"'.format(crab_log),shell=True)
das_string = subprocess.check_output('cat {} | grep "config.Data.inputDataset"'.format(crab_log),shell=True)

if n_str == "":
    print("Couldn't find 'Number of events read' in {}, please check if crab report has been run".format(crab_log))
    exit(1)

# manipulate output in order to get das string only
das_string = das_string.strip("\n").replace("'","").replace(" ","").split("=")[-1]

# manipulate output in order to get number of events only
n = int(n_str.strip('\n').replace(" ","").split(":")[-1])

# write metadata in a json file
update_json_cfg(das_string,file_list,input_miniAOD_cfg,out_cfg_full_path,n)

print("{} file updated".format(out_cfg_full_path))
