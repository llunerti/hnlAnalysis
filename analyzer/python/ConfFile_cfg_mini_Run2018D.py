import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '102X_dataRun2_Prompt_v15', '')## for 2018 D PROMPT RECO BPARKING

process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

inputFileName_list = [
'/store/data/Run2018D/ParkingBPH1/MINIAOD/05May2019promptD-v1/270000/4682963C-2EFF-FF4D-B234-8ED5973F70E4.root',
]

dataset_name       = str()
dataset_name_label = str()

if inputFileName_list[0].split("/")[2] == "mc":
    dataset_name = inputFileName_list[0].split("/")[4]
    dataset_name_label = dataset_name[0:dataset_name.find("_TuneCP5")]

elif inputFileName_list[0].split("/")[2] == "data":
    dataset_name_label = inputFileName_list[0].split("/")[3] + "_" + inputFileName_list[0].split("/")[4] 

outputFileName = 'hnlAnalyzer_'+dataset_name_label+'_tree.root'

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
                          fileName             = cms.untracked.string(outputFileName),
                          useDisplacedMuons    = cms.untracked.bool(False)
                          )

process.TFileService = cms.Service("TFileService",
       fileName = cms.string(outputFileName)
)

process.mySequence = cms.Sequence(process.demo)

process.p = cms.Path(process.mySequence)
process.schedule = cms.Schedule(process.p)
