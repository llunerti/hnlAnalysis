import json
import sys
import os
import subprocess
from datetime import datetime

def update_json_cfg(das_string,output_file_name_list,input_cfg_json,output_cfg_full_path,max_events):
    production_time = str(datetime.now())
    
    out_cfg = {}
    with open(output_cfg_full_path, "r") as f:
        out_cfg = json.loads(f.read())

    dataset_dic = {}
    dataset_dic["file_name_list"]   = output_file_name_list
    dataset_dic["production_time"]  = production_time
    dataset_dic["processed_events"] = max_events
    cat = str(input_cfg_json[das_string]["dataset_category"])
    dataset_dic["dataset_category"] = cat
    dataset_dic["das_string"] = das_string

    if cat!="data":
        dataset_dic["cross_section"]     = input_cfg_json[das_string]["cross_section"]
        dataset_dic["filter_efficiency"] = input_cfg_json[das_string]["filter_efficiency"]
    else:
        dataset_dic["integrated_lumi"] = -1.0

    short_name = str(input_cfg_json[das_string]["short_name"])
    out_cfg[short_name] = dataset_dic
    
    with open(output_cfg_full_path, "w") as f:
        json.dump(out_cfg,f, indent=4, sort_keys=True)
