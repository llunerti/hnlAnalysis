import json
from hnlAnalysis.analyzer.tools import *

import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_upgrade2018_realistic_v21', '')## https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
#process.options.allowUnscheduled = cms.untracked.bool(True)
process.load("FWCore.MessageLogger.MessageLogger_cfi")

max_events        = 100000
category          = "background"
#category          = "signal"
#das_string        = "/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#das_string        = "/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v4/MINIAODSIM"
#das_string        = "/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
das_string        = "/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18MiniAOD-102X_upgrade2018_realistic_v15-v3/MINIAODSIM"
#das_string        = "/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/RunIIAutumn18MiniAOD-Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/MINIAODSIM" 
in_cfg_full_path  = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer/cfg/miniAOD_input.json"
out_cfg_full_path = "/afs/cern.ch/work/l/llunerti/private/hnlTreeAnalyzer/cfg/hnl_tree_input.json"

#get metadata from input json file
input_miniAOD_cfg = {}
with open(in_cfg_full_path,'r') as f:
    input_miniAOD_cfg = json.loads(f.read())

tot_events = 0
inputFileName_list = []

#if max_events=-1 run on first file only
if max_events<0:
    file_name = input_miniAOD_cfg[category][das_string]["file_name_list"][0].encode('utf-8')
    inputFileName_list = [file_name]
    file_das_dict = json.loads(str(subprocess.check_output('dasgoclient --query='+file_name+' --json', shell=True)))
    tot_events = int(file_das_dict[0]["file"][0]["nevents"])
else:
    tot_events = max_events
    inputFileName_list = [item.encode('utf-8') for item in input_miniAOD_cfg[category][das_string]["file_name_list"]]

outputFileName     = 'hnlAnalyzer_'+das_string.split("/")[1]+'_tree.root'

# write metadata in a json file
update_json_cfg(category,das_string,outputFileName,input_miniAOD_cfg,out_cfg_full_path,max_events)


process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(max_events))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(inputFileName_list
    )
)

process.demo = cms.EDAnalyzer('hnlAnalyzer_miniAOD',
                          HLTriggerResults     = cms.InputTag("TriggerResults", "", "HLT"),
                          HLTriggerObjects     = cms.InputTag("slimmedPatTrigger"),
                          prunedGenParticleTag = cms.InputTag("prunedGenParticles"),
                          packedGenParticleTag = cms.InputTag("packedGenParticles"),
                          beamSpotTag          = cms.InputTag("offlineBeamSpot"),
                          VtxSample            = cms.InputTag("offlineSlimmedPrimaryVertices"),
                          Track                = cms.InputTag("packedPFCandidates"),
                          muons                = cms.InputTag("slimmedMuons"),
                          displacedMuons       = cms.InputTag("displacedStandAloneMuons"),
                          lostTracks           = cms.InputTag("lostTracks"),
                          PUInfoTag            = cms.InputTag("slimmedAddPileupInfo"),
                          fileName             = cms.untracked.string(outputFileName),
                          useDisplacedMuons    = cms.untracked.bool(False)
                          )

process.TFileService = cms.Service("TFileService",
       fileName = cms.string(outputFileName)
)

process.mySequence = cms.Sequence(process.demo)

process.p = cms.Path(process.mySequence)
process.schedule = cms.Schedule(process.p)
