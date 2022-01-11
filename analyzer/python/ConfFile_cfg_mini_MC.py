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

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

inputFileName_list = [
'/store/mc/RunIIAutumn18MiniAOD/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v4/10000/CDB6209D-2C2B-E744-8732-23BDBAB58C99.root',
#'/store/mc/RunIIAutumn18MiniAOD/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/MINIAODSIM/102X_upgrade2018_realistic_v15-v3/100000/0BF89559-F5F7-4D41-A1B6-4037F80E9A4A.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/260000/4C66BDE1-E0CE-7F46-963A-0E277B55ECDA.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau100p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/100000/0BF1EF52-5382-0042-80F9-60DCBA294BC0.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p0_ctau1000p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/100000/0E2113A5-E431-1D44-B409-97F84040365D.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/260000/3980B19C-FDB6-1844-B94D-AAFDAE71BD73.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau100p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/110000/145E2D84-1E57-F047-98C6-AED228807B0E.root',
#'/store/mc/RunIIAutumn18MiniAOD/BToNMuX_NToEMuPi_SoftQCD_b_mN1p5_ctau1000p0mm_TuneCP5_13TeV-pythia8-evtgen/MINIAODSIM/Custom_RDStar_BParking_102X_upgrade2018_realistic_v15-v2/100000/01357CC3-6E9F-C041-BEE2-0F2FE82FFA5E.root'
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
