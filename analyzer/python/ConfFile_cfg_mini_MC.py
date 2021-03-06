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
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_upgrade2018_realistic_v21', '')## https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVAnalysisSummaryTable

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# setup 'analysis'  options
options = VarParsing.VarParsing ('analysis')

# setup any defaults you want
options.inputFiles= '/store/mc/RunIIAutumn18MiniAOD/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v4/110000/B8C204C2-97BC-7E49-8C89-35A24B1C3F26.root'
options.outputFile = 'hnlAnalyzer_'+options.inputFiles[0].split("/")[4]+'_tree.root'
options.maxEvents = 10

# get and parse the command line arguments
options.parseArguments()

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(options.inputFiles
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
                          fileName             = cms.untracked.string(options.outputFile),
                          useDisplacedMuons    = cms.untracked.bool(False)
                          )

process.TFileService = cms.Service("TFileService",
       fileName = cms.string(options.outputFile)
)

process.mySequence = cms.Sequence(process.demo)

process.p = cms.Path(process.mySequence)
process.schedule = cms.Schedule(process.p)
