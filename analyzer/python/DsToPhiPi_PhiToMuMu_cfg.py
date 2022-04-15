import json
from hnlAnalysis.analyzer.tools import *

import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# setup 'analysis'  options
options = VarParsing.VarParsing ('analysis')

options.register ('globalTag',
                  '102X_upgrade2018_realistic_v21', 
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "Global tag")

options.register ('isSignal',
                  0, 
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.int,
                  "1 when processing signal samples")

# setup any defaults you want
options.inputFiles= '/store/mc/RunIIAutumn18MiniAOD/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v4/110000/B8C204C2-97BC-7E49-8C89-35A24B1C3F26.root'
options.outputFile = 'DsToPhiPi_PhiToMuMu_analyzer_'+options.inputFiles[0].split("/")[4]+'_tree.root'
options.maxEvents = -1

# get and parse the command line arguments
options.parseArguments()

process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')## https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

trigger_paths = [
  "HLT_Mu7_IP4_part*" ,      # 0
  "HLT_Mu8_IP3_part*" ,      # 1
  "HLT_Mu8_IP3p5_part*" ,    # 2  
  "HLT_Mu8_IP5_part*" ,      # 3
  "HLT_Mu8_IP6_part*" ,      # 4  
  "HLT_Mu9_IP4_part*" ,      # 5
  "HLT_Mu9_IP5_part*" ,      # 6  
  "HLT_Mu9_IP6_part*" ,      # 7  
  "HLT_Mu10p5_IP3p5_part*",  # 8
  "HLT_Mu12_IP6_part*"]      # 9

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles
    )
)

process.demo = cms.EDAnalyzer('DsToPhiPi_PhiToMuMu_miniAOD',
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
                          mu_pt_cut            = cms.untracked.double(0.3),
                          mu_eta_cut           = cms.untracked.double(2.4),
                          trigMu_pt_cut        = cms.untracked.double(6.5),
                          trigMu_eta_cut       = cms.untracked.double(1.55),
                          pi_pt_cut            = cms.untracked.double(0.5),
                          pi_eta_cut           = cms.untracked.double(2.4),
                          b_mass_cut           = cms.untracked.double(10.0),
                          mupi_mass_high_cut   = cms.untracked.double(7.0),
                          mupi_mass_low_cut    = cms.untracked.double(0.2),
                          mupi_pt_cut          = cms.untracked.double(1.0),
                          vtx_prob_cut         = cms.untracked.double(0.01),
                          is_signal            = cms.untracked.int32(options.isSignal),
                          TriggerPaths         = cms.untracked.vstring(trigger_paths),
                          fileName             = cms.untracked.string(options.outputFile)
                          )

process.TFileService = cms.Service("TFileService",
       fileName = cms.string(options.outputFile)
)

from Configuration.DataProcessing.Utils import addMonitoring
process = addMonitoring(process)

process.mySequence = cms.Sequence(process.demo)

process.p = cms.Path(process.mySequence)
process.schedule = cms.Schedule(process.p)
