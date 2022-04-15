/// framework
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
/// triggers
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
/// tracks
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// MiniAOD
#include "DataFormats/PatCandidates/interface/PackedCandidate.h" // for miniAOD
#include "DataFormats/TrackReco/interface/Track.h" // for miniAOD

/// muons
#include "DataFormats/PatCandidates/interface/Muon.h"

/// recoVertex fits
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//// gen ??
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TFile.h"
#include "TTree.h"

#include <vector>
#include "TLorentzVector.h"
#include <string>
#include "my_pdg.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// class declaration
//
class DsToPhiPi_PhiToMuMu_prompt_miniAOD : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
  public:
    explicit DsToPhiPi_PhiToMuMu_prompt_miniAOD(const edm::ParameterSet&);
    ~DsToPhiPi_PhiToMuMu_prompt_miniAOD();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    int getMatchedGenPartIdx (double pt, double eta, double phi, int pdg_id, std::vector<pat::PackedGenParticle> gen_particles);
    template <typename A, typename B> bool IsTheSame(const A& cand1, const B& cand2);


  private:
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    // ----------member data ---------------------------
    edm::EDGetTokenT<edm::TriggerResults> hlTriggerResults_;
    edm::EDGetTokenT<pat::TriggerObjectStandAloneCollection> hlTriggerObjects_;
    edm::EDGetTokenT<reco::GenParticleCollection> prunedGenToken_;
    edm::EDGetTokenT<std::vector<pat::PackedGenParticle>> packedGenToken_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
    edm::EDGetTokenT<reco::VertexCollection> vtxSample;
    edm::EDGetTokenT<std::vector<pat::PackedCandidate>> tracks_;
    //edm::EDGetTokenT<std::vector<pat::PackedCandidate>> lostTracks_;
    edm::EDGetTokenT<std::vector<pat::Muon>> muons_;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>> PUInfoToken_;

    double mu_pt_cut_;
    double mu_eta_cut_;
    double trigMu_pt_cut_;
    double trigMu_eta_cut_;
    double pi_pt_cut_;
    double pi_eta_cut_;
    double mumupi_mass_cut_;
    double mupi_mass_high_cut_;
    double mupi_mass_low_cut_;
    double mupi_pt_cut_;
    double vtx_prob_cut_;
    //int isSignal;

    std::vector<std::string> TriggerPaths;

    Int_t N_written_events;

    std::vector<float> *C_Ds_vertex_prob;
    std::vector<float> *C_Ds_mass;
    std::vector<float> *C_Ds_preFit_mass;
    std::vector<float> *C_Ds_px, *C_Ds_py, *C_Ds_pz;
    std::vector<float> *C_Ds_pt;
    std::vector<float> *C_Ds_vertex_x, *C_Ds_vertex_y, *C_Ds_vertex_z;
    std::vector<float> *C_Ds_vertex_xErr, *C_Ds_vertex_yErr, *C_Ds_vertex_zErr;
    std::vector<float> *C_Ds_vertex_sig;
    std::vector<float> *C_Ds_vertex_cos3D;
    std::vector<float> *C_Ds_vertex_cos2D;

    std::vector<float> *C_Phi_vertex_prob;
    std::vector<float> *C_Phi_mass;
    std::vector<float> *C_Phi_preFit_mass;
    std::vector<float> *C_Phi_px, *C_Phi_py, *C_Phi_pz;

    std::vector<float> *C_mu2_Phi_px, *C_mu2_Phi_py, *C_mu2_Phi_pz;
    std::vector<float> *C_mu2_Phi_pt;
    std::vector<float> *C_mu2_Phi_eta;
    std::vector<float> *C_mu2_Phi_BS_ips_xy, *C_mu2_Phi_PV_ips_z;
    std::vector<float> *C_mu2_Phi_BS_ips;
    std::vector<float> *C_mu2_Phi_BS_ip_xy, *C_mu2_Phi_PV_ip_z;
    std::vector<float> *C_mu2_Phi_BS_ip;
    std::vector<int>   *C_mu2_Phi_charge;
    std::vector<short> *C_mu2_Phi_isSoft;
    std::vector<short> *C_mu2_Phi_isLoose;
    std::vector<short> *C_mu2_Phi_isMedium;
    std::vector<unsigned> *C_mu2_Phi_idx;

    std::vector<float> *C_mu1_Phi_px, *C_mu1_Phi_py, *C_mu1_Phi_pz;
    std::vector<float> *C_mu1_Phi_pt;
    std::vector<float> *C_mu1_Phi_eta;
    std::vector<float> *C_mu1_Phi_BS_ips_xy, *C_mu1_Phi_PV_ips_z;
    std::vector<float> *C_mu1_Phi_BS_ips;
    std::vector<float> *C_mu1_Phi_BS_ip_xy, *C_mu1_Phi_PV_ip_z;
    std::vector<float> *C_mu1_Phi_BS_ip;
    std::vector<int>   *C_mu1_Phi_charge;
    std::vector<short> *C_mu1_Phi_isSoft;
    std::vector<short> *C_mu1_Phi_isLoose;
    std::vector<short> *C_mu1_Phi_isMedium;
    std::vector<unsigned> *C_mu1_Phi_idx;

    std::vector<unsigned> *C_mu_trig_idx;
    std::vector<float> *C_mu_trig_pt;
    std::vector<float> *C_mu_trig_eta;
    std::vector<float> *C_mu_trig_phi;

    std::vector<int>   *C_pi_charge;
    std::vector<float> *C_pi_px, *C_pi_py, *C_pi_pz;
    std::vector<float> *C_pi_pt;
    std::vector<float> *C_pi_eta;
    std::vector<float> *C_pi_BS_ips_xy, *C_pi_BS_ips_z;
    std::vector<float> *C_pi_BS_ip_xy, *C_pi_BS_ip_z;

    std::vector<float> *PV_x, *PV_y, *PV_z;
    std::vector<float> *PV_xErr, *PV_yErr, *PV_zErr;
    std::vector<float> *PV_prob;
    //std::vector<int>   *PV_dN;

    std::vector<short>  *HLT_mu7_ip4_matched;
    std::vector<short>  *HLT_mu8_ip3_matched;
    std::vector<short>  *HLT_mu8_ip3p5_matched ;
    std::vector<short>  *HLT_mu8_ip5_matched;
    std::vector<short>  *HLT_mu8_ip6_matched;
    std::vector<short>  *HLT_mu9_ip4_matched;
    std::vector<short>  *HLT_mu9_ip5_matched;
    std::vector<short>  *HLT_mu9_ip6_matched;
    std::vector<short>  *HLT_mu10p5_ip3p5_matched;
    std::vector<short>  *HLT_mu12_ip6_matched;

    std::vector<float>  *HLT_mu7_ip4_eta;
    std::vector<float>  *HLT_mu8_ip3_eta;
    std::vector<float>  *HLT_mu8_ip3p5_eta;
    std::vector<float>  *HLT_mu8_ip5_eta;
    std::vector<float>  *HLT_mu8_ip6_eta;
    std::vector<float>  *HLT_mu9_ip4_eta;
    std::vector<float>  *HLT_mu9_ip5_eta;
    std::vector<float>  *HLT_mu9_ip6_eta;
    std::vector<float>  *HLT_mu10p5_ip3p5_eta;
    std::vector<float>  *HLT_mu12_ip6_eta;

    std::vector<float>  *HLT_mu7_ip4_pt;
    std::vector<float>  *HLT_mu8_ip3_pt;
    std::vector<float>  *HLT_mu8_ip3p5_pt ;
    std::vector<float>  *HLT_mu8_ip5_pt;
    std::vector<float>  *HLT_mu8_ip6_pt;
    std::vector<float>  *HLT_mu9_ip4_pt;
    std::vector<float>  *HLT_mu9_ip5_pt;
    std::vector<float>  *HLT_mu9_ip6_pt;
    std::vector<float>  *HLT_mu10p5_ip3p5_pt;
    std::vector<float>  *HLT_mu12_ip6_pt;

    std::vector<float>  *HLT_mu7_ip4_dr;
    std::vector<float>  *HLT_mu8_ip3_dr;
    std::vector<float>  *HLT_mu8_ip3p5_dr ;
    std::vector<float>  *HLT_mu8_ip5_dr;
    std::vector<float>  *HLT_mu8_ip6_dr;
    std::vector<float>  *HLT_mu9_ip4_dr;
    std::vector<float>  *HLT_mu9_ip5_dr;
    std::vector<float>  *HLT_mu9_ip6_dr;
    std::vector<float>  *HLT_mu10p5_ip3p5_dr;
    std::vector<float>  *HLT_mu12_ip6_dr;

    std::vector<short>  *HLT_mu7_ip4_fired;
    std::vector<short>  *HLT_mu8_ip3_fired;
    std::vector<short>  *HLT_mu8_ip3p5_fired;
    std::vector<short>  *HLT_mu8_ip5_fired;
    std::vector<short>  *HLT_mu8_ip6_fired;
    std::vector<short>  *HLT_mu9_ip4_fired;
    std::vector<short>  *HLT_mu9_ip5_fired;
    std::vector<short>  *HLT_mu9_ip6_fired;
    std::vector<short>  *HLT_mu10p5_ip3p5_fired;
    std::vector<short>  *HLT_mu12_ip6_fired;


    Int_t nCand;

    Int_t run;
    Int_t event;
    Float_t lumi;

    Int_t nPV;
    Int_t nPU_trueInt;
    Int_t numTrack;
    Int_t numMuon;

    TTree *wwtree;
    TFile *f;
    std::string fileName;
    my_pdg pdg;

};


DsToPhiPi_PhiToMuMu_prompt_miniAOD::DsToPhiPi_PhiToMuMu_prompt_miniAOD(const edm::ParameterSet& iConfig) :
  hlTriggerResults_ (consumes<edm::TriggerResults> (iConfig.getParameter<edm::InputTag>("HLTriggerResults"))),
  hlTriggerObjects_ (consumes<pat::TriggerObjectStandAloneCollection> (iConfig.getParameter<edm::InputTag>("HLTriggerObjects"))),
  prunedGenToken_   (consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("prunedGenParticleTag"))),
  packedGenToken_   (consumes<std::vector<pat::PackedGenParticle>>(iConfig.getParameter<edm::InputTag>("packedGenParticleTag"))),
  beamSpotToken_    (consumes<reco::BeamSpot> (iConfig.getParameter<edm::InputTag>("beamSpotTag"))),
  vtxSample         (consumes<reco::VertexCollection> (iConfig.getParameter<edm::InputTag>("VtxSample"))),
  tracks_           (consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("Track"))),
  //lostTracks_       (consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("lostTracks"))),
  muons_            (consumes<std::vector<pat::Muon>> (iConfig.getParameter<edm::InputTag>("muons"))),
  PUInfoToken_      (consumes<std::vector<PileupSummaryInfo>> (iConfig.getParameter<edm::InputTag>("PUInfoTag"))),
  mu_pt_cut_        (iConfig.getUntrackedParameter<double>("mu_pt_cut" )),
  mu_eta_cut_       (iConfig.getUntrackedParameter<double>("mu_eta_cut")),
  trigMu_pt_cut_    (iConfig.getUntrackedParameter<double>("trigMu_pt_cut" )),
  trigMu_eta_cut_   (iConfig.getUntrackedParameter<double>("trigMu_eta_cut")),
  pi_pt_cut_        (iConfig.getUntrackedParameter<double>("pi_pt_cut" )),
  pi_eta_cut_       (iConfig.getUntrackedParameter<double>("pi_eta_cut")),
  mumupi_mass_cut_  (iConfig.getUntrackedParameter<double>("b_mass_cut")),
  mupi_mass_high_cut_(iConfig.getUntrackedParameter<double>("mupi_mass_high_cut")),
  mupi_mass_low_cut_(iConfig.getUntrackedParameter<double>("mupi_mass_low_cut")),
  mupi_pt_cut_      (iConfig.getUntrackedParameter<double>("mupi_pt_cut")),
  vtx_prob_cut_     (iConfig.getUntrackedParameter<double>("vtx_prob_cut")),
  //isSignal           (iConfig.getUntrackedParameter<int>("is_signal")),
  TriggerPaths       (iConfig.getUntrackedParameter<std::vector<std::string>>("TriggerPaths")),

  C_Ds_vertex_prob(0),
  C_Ds_mass(0),
  C_Ds_preFit_mass(0),
  C_Ds_px(0),
  C_Ds_py(0),  
  C_Ds_pz(0),
  C_Ds_pt(0),
  C_Ds_vertex_x(0),  
  C_Ds_vertex_y(0), 
  C_Ds_vertex_z(0),
  C_Ds_vertex_xErr(0),  
  C_Ds_vertex_yErr(0), 
  C_Ds_vertex_zErr(0),
  C_Ds_vertex_sig(0),
  C_Ds_vertex_cos3D(0),
  C_Ds_vertex_cos2D(0),

  C_Phi_vertex_prob(0),
  C_Phi_mass(0),
  C_Phi_preFit_mass(0),
  C_Phi_px(0),
  C_Phi_py(0),  
  C_Phi_pz(0),

  C_mu2_Phi_px(0),
  C_mu2_Phi_py(0),       
  C_mu2_Phi_pz(0),
  C_mu2_Phi_pt(0),
  C_mu2_Phi_eta(0),
  C_mu2_Phi_BS_ips_xy(0),
  C_mu2_Phi_PV_ips_z(0),
  C_mu2_Phi_BS_ips(0),
  C_mu2_Phi_BS_ip_xy(0),
  C_mu2_Phi_PV_ip_z(0),
  C_mu2_Phi_BS_ip(0),
  C_mu2_Phi_charge(0),
  C_mu2_Phi_isSoft(0),
  C_mu2_Phi_isLoose(0),
  C_mu2_Phi_isMedium(0),
  C_mu2_Phi_idx(0),

  C_mu1_Phi_px(0),
  C_mu1_Phi_py(0),
  C_mu1_Phi_pz(0),
  C_mu1_Phi_pt(0),
  C_mu1_Phi_eta(0),
  C_mu1_Phi_BS_ips_xy(0),
  C_mu1_Phi_PV_ips_z(0),
  C_mu1_Phi_BS_ips(0),
  C_mu1_Phi_BS_ip_xy(0),
  C_mu1_Phi_PV_ip_z(0),
  C_mu1_Phi_BS_ip(0),
  C_mu1_Phi_charge(0),
  C_mu1_Phi_isSoft(0),
  C_mu1_Phi_isLoose(0),
  C_mu1_Phi_isMedium(0),
  C_mu1_Phi_idx(0),

  C_mu_trig_idx(0),
  C_mu_trig_pt(0),
  C_mu_trig_eta(0),
  C_mu_trig_phi(0),

  C_pi_charge(0),
  C_pi_px(0),
  C_pi_py(0),
  C_pi_pz(0),
  C_pi_pt(0),
  C_pi_eta(0),
  C_pi_BS_ips_xy(0),
  C_pi_BS_ips_z(0),
  C_pi_BS_ip_xy(0),
  C_pi_BS_ip_z(0),

  PV_x(0)  , PV_y(0), PV_z(0),
  PV_xErr(0)  , PV_yErr(0), PV_zErr(0),
  PV_prob(0)  , //PV_dN(0),

  HLT_mu7_ip4_matched(0),
  HLT_mu8_ip3_matched(0),
  HLT_mu8_ip3p5_matched (0),
  HLT_mu8_ip5_matched(0),
  HLT_mu8_ip6_matched(0),
  HLT_mu9_ip4_matched(0),
  HLT_mu9_ip5_matched(0),
  HLT_mu9_ip6_matched(0),
  HLT_mu10p5_ip3p5_matched(0),
  HLT_mu12_ip6_matched(0),

  HLT_mu7_ip4_eta(0),
  HLT_mu8_ip3_eta(0),
  HLT_mu8_ip3p5_eta (0),
  HLT_mu8_ip5_eta(0),
  HLT_mu8_ip6_eta(0),
  HLT_mu9_ip4_eta(0),
  HLT_mu9_ip5_eta(0),
  HLT_mu9_ip6_eta(0),
  HLT_mu10p5_ip3p5_eta(0),
  HLT_mu12_ip6_eta(0),

  HLT_mu7_ip4_pt(0),
  HLT_mu8_ip3_pt(0),
  HLT_mu8_ip3p5_pt (0),
  HLT_mu8_ip5_pt(0),
  HLT_mu8_ip6_pt(0),
  HLT_mu9_ip4_pt(0),
  HLT_mu9_ip5_pt(0),
  HLT_mu9_ip6_pt(0),
  HLT_mu10p5_ip3p5_pt(0),
  HLT_mu12_ip6_pt(0),

  HLT_mu7_ip4_dr(0),
  HLT_mu8_ip3_dr(0),
  HLT_mu8_ip3p5_dr (0),
  HLT_mu8_ip5_dr(0),
  HLT_mu8_ip6_dr(0),
  HLT_mu9_ip4_dr(0),
  HLT_mu9_ip5_dr(0),
  HLT_mu9_ip6_dr(0),
  HLT_mu10p5_ip3p5_dr(0),
  HLT_mu12_ip6_dr(0),

  HLT_mu7_ip4_fired(0),
  HLT_mu8_ip3_fired(0),
  HLT_mu8_ip3p5_fired (0),
  HLT_mu8_ip5_fired(0),
  HLT_mu8_ip6_fired(0),
  HLT_mu9_ip4_fired(0),
  HLT_mu9_ip5_fired(0),
  HLT_mu9_ip6_fired(0),
  HLT_mu10p5_ip3p5_fired(0),
  HLT_mu12_ip6_fired(0),

  nCand(0),

  run(0),
  event(0),
  lumi(0),

  nPV(0),
  nPU_trueInt(0),
  numTrack(0)
{
  fileName = iConfig.getUntrackedParameter<std::string>("fileName","hnlAnalyzer_2018BPark-MiniAOD.root");
  usesResource("TFileService");
  N_written_events = 0;
  pdg = my_pdg();

}


DsToPhiPi_PhiToMuMu_prompt_miniAOD::~DsToPhiPi_PhiToMuMu_prompt_miniAOD()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
  void
DsToPhiPi_PhiToMuMu_prompt_miniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  run   = iEvent.id().run();
  event = iEvent.id().event();
  lumi  = iEvent.luminosityBlock();

  edm::ESHandle<TransientTrackBuilder> TTrackBuilder; 
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder",TTrackBuilder); 

  edm::ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);

  edm::Handle<edm::TriggerResults> triggerResults_handle;
  iEvent.getByToken(hlTriggerResults_, triggerResults_handle);

  edm::Handle<pat::TriggerObjectStandAloneCollection> triggerObjects_handle;
  iEvent.getByToken(hlTriggerObjects_, triggerObjects_handle);

  edm::Handle<reco::GenParticleCollection> prunedGenParticleCollection;
  iEvent.getByToken(prunedGenToken_,prunedGenParticleCollection);

  edm::Handle<std::vector<pat::PackedGenParticle>> packedGenParticleCollection;
  iEvent.getByToken(packedGenToken_,packedGenParticleCollection);

  edm::Handle<reco::BeamSpot> theBeamSpot;
  iEvent.getByToken(beamSpotToken_,theBeamSpot);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(vtxSample, recVtxs);

  reco::Vertex thePrimaryV;
  thePrimaryV = reco::Vertex(*(recVtxs->begin()));
  const reco::VertexCollection & vertices = *recVtxs.product();

  edm::Handle <std::vector<pat::Muon>>thePATMuonHandle;
  iEvent.getByToken(muons_, thePATMuonHandle);

  edm::Handle<std::vector<pat::PackedCandidate> >thePATTrackHandle;
  iEvent.getByToken(tracks_, thePATTrackHandle);

  edm::Handle<std::vector<PileupSummaryInfo>>  PupInfoHandle;

  nPV      = vertices.size();
  numTrack = thePATTrackHandle->size();
  numMuon  = thePATMuonHandle->size();

  if (iEvent.getByToken(PUInfoToken_, PupInfoHandle)){
    for (std::vector<PileupSummaryInfo>::const_iterator pui = PupInfoHandle->begin(); pui != PupInfoHandle->end(); ++pui){
      int BX = pui->getBunchCrossing();
      if(BX == 0) {
	nPU_trueInt = pui->getTrueNumInteractions();
      }
    }
  }
  else
    nPU_trueInt = -1;

  //Save PV info
  for(unsigned i=0; i<(*recVtxs).size(); ++i){
    const reco::Vertex* pv = &(*recVtxs).at(i);
    Double_t vertex_x    = pv->x();
    Double_t vertex_y    = pv->y();
    Double_t vertex_z    = pv->z();
    Double_t vertex_xErr = pv->covariance(0, 0);
    Double_t vertex_yErr = pv->covariance(1, 1);
    Double_t vertex_zErr = pv->covariance(2, 2);
    Double_t vertex_prob = (TMath::Prob(pv->chi2(), (int) pv->ndof()));

    PV_x ->push_back(vertex_x);
    PV_y ->push_back(vertex_y);
    PV_z ->push_back(vertex_z);
    PV_xErr ->push_back(vertex_xErr);
    PV_yErr ->push_back(vertex_yErr);
    PV_zErr ->push_back(vertex_zErr);
    PV_prob ->push_back(vertex_prob);
  }

  unsigned int nTrigPaths = (unsigned int)TriggerPaths.size();

  std::vector<short> TriggersFired(nTrigPaths);
  std::vector<short> TriggerMatches(nTrigPaths);
  std::vector<float> TriggerPathPt(nTrigPaths);
  std::vector<float> TriggerPathEta(nTrigPaths);
  std::vector<float> TriggerPathDR(nTrigPaths);

  // Do I really need information about firing trigger?
  //
  /*
  if (triggerResults_handle.isValid())
  {
    const edm::TriggerNames & TheTriggerNames = iEvent.triggerNames(*triggerResults_handle);
    std::vector<int> part_bpark = {0,1,2,3,4,5};

    for(unsigned i=0; i<nTrigPaths; ++i){
      for (int part : part_bpark){
	std::string trigger_path = TriggerPaths.at(i).substr(0,TriggerPaths.at(i).find("*"));
	std::string trigger_path_addPart; 
	trigger_path_addPart = trigger_path + std::to_string(part);

	for (int version = 1; version < 30; version++){
	  std::string full_trigger_path; 
	  full_trigger_path = trigger_path_addPart + "_v" + std::to_string(version);
	  unsigned int bit = TheTriggerNames.triggerIndex(edm::InputTag(full_trigger_path).label());

	  if ((bit < triggerResults_handle->size()) && (triggerResults_handle->accept(bit)) && (!triggerResults_handle->error(bit)))
	    TriggersFired[i] = 1;
	}
      }
    }

    HLT_mu7_ip4_fired->push_back(TriggersFired[0]);
    HLT_mu8_ip3_fired->push_back(TriggersFired[1]);
    HLT_mu8_ip3p5_fired ->push_back(TriggersFired[2]);
    HLT_mu8_ip5_fired->push_back(TriggersFired[3]);
    HLT_mu8_ip6_fired->push_back(TriggersFired[4]);
    HLT_mu9_ip4_fired->push_back(TriggersFired[5]);
    HLT_mu9_ip5_fired->push_back(TriggersFired[6]);
    HLT_mu9_ip6_fired->push_back(TriggersFired[7]);
    HLT_mu10p5_ip3p5_fired->push_back(TriggersFired[8]);
    HLT_mu12_ip6_fired->push_back(TriggersFired[9]);
  }
  else
  {
    std::cout << " No trigger Results in event :( " << run << "," << event << std::endl;
  }
  */

  for ( unsigned i_trigmu=0; i_trigmu<thePATMuonHandle->size(); ++i_trigmu){

    const pat::Muon* iTrigMu = &(*thePATMuonHandle).at(i_trigmu);

    //cut on muon pt and eta
    if(iTrigMu->pt() < trigMu_pt_cut_) continue;
    if(std::abs(iTrigMu->eta()) > trigMu_eta_cut_)  continue;

    if (!iTrigMu->isSoftMuon(thePrimaryV)) continue;

    for (unsigned i = 0; i < nTrigPaths; ++i) {

      bool match = false;
      float best_matching_path_dr  = -9999.;
      float best_matching_path_pt  = -9999.;
      float best_matching_path_eta = -9999.;
      float min_dr = 9999.;

      if(iTrigMu->triggerObjectMatches().size()!=0){

	//loop over trigger object matched to muon
	for(size_t j=0; j<iTrigMu->triggerObjectMatches().size();j++){

	  if(iTrigMu->triggerObjectMatch(j)!=0 && iTrigMu->triggerObjectMatch(j)->hasPathName(TriggerPaths[i],true,true)){

	    float trig_dr  = reco::deltaR(iTrigMu->triggerObjectMatch(j)->p4(), iTrigMu->p4()); 
	    float trig_pt  = iTrigMu->triggerObjectMatch(j)->pt();                   
	    float trig_eta = iTrigMu->triggerObjectMatch(j)->eta();                   

	    //select the match with smallest dR
	    if (trig_dr<min_dr){
	      match = true;
	      min_dr = trig_dr;
	      best_matching_path_dr  = trig_dr;
	      best_matching_path_pt  = trig_pt;
	      best_matching_path_eta = trig_eta;
	    }
	  }
	}
      }

      TriggerMatches[i] = match? 1: 0;
      TriggerPathDR[i]  = best_matching_path_dr;
      TriggerPathPt[i]  = best_matching_path_pt;
      TriggerPathEta[i] = best_matching_path_eta;

    }

    C_mu_trig_idx ->push_back(i_trigmu);
    C_mu_trig_pt ->push_back(iTrigMu->pt());
    C_mu_trig_eta ->push_back(iTrigMu->eta());
    C_mu_trig_phi ->push_back(iTrigMu->phi());

    HLT_mu7_ip4_matched->push_back(TriggerMatches[0]);
    HLT_mu8_ip3_matched->push_back(TriggerMatches[1]);
    HLT_mu8_ip3p5_matched ->push_back(TriggerMatches[2]);
    HLT_mu8_ip5_matched->push_back(TriggerMatches[3]);
    HLT_mu8_ip6_matched->push_back(TriggerMatches[4]);
    HLT_mu9_ip4_matched->push_back(TriggerMatches[5]);
    HLT_mu9_ip5_matched->push_back(TriggerMatches[6]);
    HLT_mu9_ip6_matched->push_back(TriggerMatches[7]);
    HLT_mu10p5_ip3p5_matched->push_back(TriggerMatches[8]);
    HLT_mu12_ip6_matched->push_back(TriggerMatches[9]);

    HLT_mu7_ip4_eta->push_back(TriggerPathEta[0]);
    HLT_mu8_ip3_eta->push_back(TriggerPathEta[1]);
    HLT_mu8_ip3p5_eta ->push_back(TriggerPathEta[2]);
    HLT_mu8_ip5_eta->push_back(TriggerPathEta[3]);
    HLT_mu8_ip6_eta->push_back(TriggerPathEta[4]);
    HLT_mu9_ip4_eta->push_back(TriggerPathEta[5]);
    HLT_mu9_ip5_eta->push_back(TriggerPathEta[6]);
    HLT_mu9_ip6_eta->push_back(TriggerPathEta[7]);
    HLT_mu10p5_ip3p5_eta->push_back(TriggerPathEta[8]);
    HLT_mu12_ip6_eta->push_back(TriggerPathEta[9]);

    HLT_mu7_ip4_pt->push_back(TriggerPathPt[0]);
    HLT_mu8_ip3_pt->push_back(TriggerPathPt[1]);
    HLT_mu8_ip3p5_pt ->push_back(TriggerPathPt[2]);
    HLT_mu8_ip5_pt->push_back(TriggerPathPt[3]);
    HLT_mu8_ip6_pt->push_back(TriggerPathPt[4]);
    HLT_mu9_ip4_pt->push_back(TriggerPathPt[5]);
    HLT_mu9_ip5_pt->push_back(TriggerPathPt[6]);
    HLT_mu9_ip6_pt->push_back(TriggerPathPt[7]);
    HLT_mu10p5_ip3p5_pt->push_back(TriggerPathPt[8]);
    HLT_mu12_ip6_pt->push_back(TriggerPathPt[9]);

    HLT_mu7_ip4_dr->push_back(TriggerPathDR[0]);
    HLT_mu8_ip3_dr->push_back(TriggerPathDR[1]);
    HLT_mu8_ip3p5_dr ->push_back(TriggerPathDR[2]);
    HLT_mu8_ip5_dr->push_back(TriggerPathDR[3]);
    HLT_mu8_ip6_dr->push_back(TriggerPathDR[4]);
    HLT_mu9_ip4_dr->push_back(TriggerPathDR[5]);
    HLT_mu9_ip5_dr->push_back(TriggerPathDR[6]);
    HLT_mu9_ip6_dr->push_back(TriggerPathDR[7]);
    HLT_mu10p5_ip3p5_dr->push_back(TriggerPathDR[8]);
    HLT_mu12_ip6_dr->push_back(TriggerPathDR[9]);
  }

  nCand= 0;
  KinematicParticleFactoryFromTransientTrack pFactory;

    for ( unsigned i_muphi1=0; i_muphi1<thePATMuonHandle->size(); ++i_muphi1){

      const pat::Muon* iMuonPhi1 = &(*thePATMuonHandle).at(i_muphi1);

      //cut on muon pt and eta
      if(iMuonPhi1->pt() < mu_pt_cut_) continue;
      if(std::abs(iMuonPhi1->eta()) > mu_eta_cut_)  continue;

      //save muon id info
      bool isSoftMuonPhi1 = false;
      bool isLooseMuonPhi1 = false;
      bool isMediumMuonPhi1 = false;

      if (iMuonPhi1->isSoftMuon(thePrimaryV)) isSoftMuonPhi1=true;
      if (iMuonPhi1->isLooseMuon())           isLooseMuonPhi1=true;
      if (iMuonPhi1->isMediumMuon())          isMediumMuonPhi1=true;


      TrackRef inTrackMuPhi1;
      inTrackMuPhi1 = iMuonPhi1->track();
      if (inTrackMuPhi1.isNull())  continue;
      if (!(inTrackMuPhi1->quality(reco::TrackBase::highPurity)))  continue;

      TLorentzVector p4muphi1;
      p4muphi1.SetPtEtaPhiM(iMuonPhi1->pt(), iMuonPhi1->eta(), iMuonPhi1->phi(), pdg.PDG_MUON_MASS);

      for (unsigned i_muphi2=i_muphi1+1; i_muphi2<thePATMuonHandle->size(); ++i_muphi2){

	const pat::Muon* iMuonPhi2 = &(*thePATMuonHandle).at(i_muphi2);

	//cuts on muon pt and eta
	if (iMuonPhi2->pt() < mu_pt_cut_) continue;
	if (std::abs(iMuonPhi2->eta()) > mu_eta_cut_)  continue;

	//save muon id info
	bool isSoftMuonPhi2 = false;
	bool isLooseMuonPhi2 = false;
	bool isMediumMuonPhi2 = false;

	if (iMuonPhi2->isSoftMuon(thePrimaryV)) isSoftMuonPhi2=true;
	if (iMuonPhi2->isLooseMuon())           isLooseMuonPhi2=true;
	if (iMuonPhi2->isMediumMuon())          isMediumMuonPhi2=true;

	//cuts on muon track
	TrackRef inTrackMuPhi2;
	inTrackMuPhi2 = iMuonPhi2->track();
	if( inTrackMuPhi2.isNull())  continue;
	if(!(inTrackMuPhi2->quality(reco::TrackBase::highPurity)))  continue;

	TransientTrack muonPhi1TT((*TTrackBuilder).build(inTrackMuPhi1));
	TransientTrack muonPhi2TT((*TTrackBuilder).build(inTrackMuPhi2));


	if(!muonPhi2TT.isValid()) continue;
	if(!muonPhi1TT.isValid()) continue;

	float muon_sigma = pdg.PDG_MUON_MASS * 1.e-6;
	float PM_sigma = 1.e-7;
	float chi = 0.;
	float ndf = 0.;

	std::vector<RefCountedKinematicParticle> phi_particles;
	phi_particles.push_back(pFactory.particle(muonPhi1TT, pdg.PM_PDG_MUON_MASS, chi, ndf, muon_sigma));
	phi_particles.push_back(pFactory.particle(muonPhi2TT, pdg.PM_PDG_MUON_MASS, chi, ndf, muon_sigma));
	KinematicParticleVertexFitter phiToMuMu_vertexFitter;
	RefCountedKinematicTree phiToMuMu_kinTree;
	phiToMuMu_kinTree = phiToMuMu_vertexFitter.fit(phi_particles);

	if (!phiToMuMu_kinTree->isValid()) continue;

	phiToMuMu_kinTree->movePointerToTheTop();
	RefCountedKinematicParticle mumu_particle = phiToMuMu_kinTree->currentParticle();
	RefCountedKinematicVertex   mumu_vtx      = phiToMuMu_kinTree->currentDecayVertex();

	double fitted_mumu_mass = mumu_particle->currentState().mass();

	if ( fitted_mumu_mass < pdg.PDG_PHI_MASS - 0.05) continue;
	if ( fitted_mumu_mass > pdg.PDG_PHI_MASS + 0.05) continue;

	double mumu_vtxprob = TMath::Prob(mumu_vtx->chiSquared(), mumu_vtx->degreesOfFreedom());
	if(mumu_vtxprob < vtx_prob_cut_) continue;

	for (std::vector<pat::PackedCandidate>::const_iterator iTrack1 = thePATTrackHandle->begin(); iTrack1 != thePATTrackHandle->end(); ++iTrack1){

	  //Nota bene: if you want to use dxy or dz you need to be sure 
	  //the pt of the tracks is bigger than 0.5 GeV, otherwise you 
	  //will get an error related to covariance matrix.
	  //Next lines are very recommended

	  if(iTrack1->pt() <= pi_pt_cut_) continue;
	  if(std::abs(iTrack1->eta()) > pi_eta_cut_) continue;
	  if(iTrack1->charge()==0) continue;// NO neutral objects
	  if(std::abs(iTrack1->pdgId())!=211) continue;//Due to the lack of the particle ID all the tracks for cms are pions(ID==211)
	  if(!(iTrack1->trackHighPurity())) continue; 

	  //cuts on mupi mass and pt
	  //TLorentzVector p4muphi2,p4pi1;
	  TLorentzVector p4pi1;
	  TLorentzVector p4muphi2;
	  p4pi1.SetPtEtaPhiM(iTrack1->pt(),iTrack1->eta(),iTrack1->phi(), pdg.PDG_PION_MASS);
	  p4muphi2.SetPtEtaPhiM(iMuonPhi2->pt(), iMuonPhi2->eta(), iMuonPhi2->phi(), pdg.PDG_MUON_MASS);

	  TransientTrack pion1TT((*TTrackBuilder).build(iTrack1->pseudoTrack()));
	  if(!pion1TT.isValid()) continue;

	  //initialize Ds->Phi(->MuMu)Pi PiKinematicParticles vector
	  std::vector <RefCountedKinematicParticle> ds_particles;
	  ds_particles.push_back(pFactory.particle(muonPhi2TT, pdg.PM_PDG_MUON_MASS, chi,ndf, muon_sigma));
	  ds_particles.push_back(pFactory.particle(muonPhi1TT, pdg.PM_PDG_MUON_MASS, chi,ndf, muon_sigma));
	  ds_particles.push_back(pFactory.particle(pion1TT, pdg.PM_PDG_PION_MASS, chi,ndf, PM_sigma)); 

	  RefCountedKinematicTree dsToPhiPi_kinTree;
	  KinematicConstrainedVertexFitter DsToPiMuMu_constrainedVertexFitter;

	  // phi mass constrained to the first two track in KinematicParticle vector (i.e. the two muons)
	  MultiTrackKinematicConstraint *ConstraintPhiMass = new TwoTrackMassKinematicConstraint(pdg.PM_PDG_PHI_MASS);

	  // fit mu mu pi constraining dimuon to phi mass
	  dsToPhiPi_kinTree = DsToPiMuMu_constrainedVertexFitter.fit(ds_particles, ConstraintPhiMass);
	  if (!dsToPhiPi_kinTree->isValid()) continue;

	  // get fitted particle and vertex
	  dsToPhiPi_kinTree->movePointerToTheTop();
	  RefCountedKinematicParticle ds_particle = dsToPhiPi_kinTree->currentParticle();
	  RefCountedKinematicVertex   ds_vtx   = dsToPhiPi_kinTree->currentDecayVertex();
	  if (!ds_vtx->vertexIsValid())  continue;

	  double fitted_ds_mass = ds_particle->currentState().mass();

	  // D_s mass = 1.97, D mass = 1.86
	  if (fitted_ds_mass < 1.75) continue; //
	  if (fitted_ds_mass > 2.15) continue; //

	  double ds_vtxprob   = TMath::Prob(ds_vtx->chiSquared(), (int) ds_vtx->degreesOfFreedom());
	  if(ds_vtxprob < vtx_prob_cut_) continue;

	  // good candidate found
	  ++nCand;

	  // get Ds vertex info
	  Double_t ds_vx    = ds_vtx->position().x();
	  Double_t ds_vy    = ds_vtx->position().y();
	  Double_t ds_vz    = ds_vtx->position().z();
	  Double_t ds_vxErr = ds_vtx->error().cxx();
	  Double_t ds_vyErr = ds_vtx->error().cyy();
	  Double_t ds_vzErr = ds_vtx->error().czz();

	  //get Ds pt
	  Double_t px_ds = ds_particle->currentState().globalMomentum().x();
	  Double_t py_ds = ds_particle->currentState().globalMomentum().y();
	  Double_t pz_ds = ds_particle->currentState().globalMomentum().z();
	  Double_t p_ds = ds_particle->currentState().globalMomentum().mag();
	  Double_t pt_ds = TMath::Sqrt(px_ds*px_ds + py_ds*py_ds);

	  //compute cos2D and cos3D wrt beam spot
	  Double_t dx_ds = ds_vtx->position().x() - (*theBeamSpot).position().x();
	  Double_t dy_ds = ds_vtx->position().y() - (*theBeamSpot).position().y();
	  Double_t dz_ds = ds_vtx->position().z() - (*theBeamSpot).position().z();
	  Double_t cos3D_ds = (px_ds*dx_ds + py_ds*dy_ds + pz_ds*dz_ds)/(sqrt(dx_ds*dx_ds + dy_ds*dy_ds + dz_ds*dz_ds)*p_ds);
	  Double_t cos2D_ds = (px_ds*dx_ds + py_ds*dy_ds)/(sqrt(dx_ds*dx_ds + dy_ds*dy_ds)*sqrt(px_ds*px_ds + py_ds*py_ds));

	  //   SAVE

	  C_Ds_vertex_prob ->push_back(ds_vtxprob);
	  C_Ds_mass ->push_back(fitted_ds_mass);
	  C_Ds_preFit_mass ->push_back((p4pi1 + p4muphi1 + p4muphi2).M());
	  C_Ds_px ->push_back(px_ds);
	  C_Ds_py ->push_back(py_ds);
	  C_Ds_pz ->push_back(pz_ds);
	  C_Ds_pt ->push_back(pt_ds);
	  C_Ds_vertex_x   ->push_back(ds_vx);
	  C_Ds_vertex_y   ->push_back(ds_vy);
	  C_Ds_vertex_z   ->push_back(ds_vz);
	  C_Ds_vertex_xErr->push_back(ds_vxErr);
	  C_Ds_vertex_yErr->push_back(ds_vyErr);
	  C_Ds_vertex_zErr->push_back(ds_vzErr);
	  C_Ds_vertex_sig->push_back(sqrt(ds_vx*ds_vx + ds_vy*ds_vy + ds_vz*ds_vz)/sqrt(ds_vxErr*ds_vxErr + ds_vyErr*ds_vyErr + ds_vzErr*ds_vzErr));
	  C_Ds_vertex_cos3D->push_back(cos3D_ds);
	  C_Ds_vertex_cos2D->push_back(cos2D_ds);

	  C_Phi_vertex_prob ->push_back(mumu_vtxprob);
	  C_Phi_mass ->push_back(fitted_mumu_mass);
	  C_Phi_preFit_mass ->push_back((p4muphi2 + p4muphi1).M());
	  C_Phi_px ->push_back(mumu_particle->currentState().globalMomentum().x());
	  C_Phi_py ->push_back(mumu_particle->currentState().globalMomentum().y());
	  C_Phi_pz ->push_back(mumu_particle->currentState().globalMomentum().z());

	  C_mu2_Phi_px ->push_back(iMuonPhi2->px());
	  C_mu2_Phi_py ->push_back(iMuonPhi2->py());
	  C_mu2_Phi_pz ->push_back(iMuonPhi2->pz());
	  C_mu2_Phi_pt ->push_back(iMuonPhi2->pt());
	  C_mu2_Phi_eta ->push_back(iMuonPhi2->eta());
	  C_mu2_Phi_BS_ips_xy ->push_back(iMuonPhi2->dB(pat::Muon::BS2D)/iMuonPhi2->edB(pat::Muon::BS2D));
	  C_mu2_Phi_BS_ips    ->push_back(iMuonPhi2->dB(pat::Muon::BS3D)/iMuonPhi2->edB(pat::Muon::BS3D));
	  C_mu2_Phi_PV_ips_z  ->push_back(iMuonPhi2->dB(pat::Muon::PVDZ)/iMuonPhi2->edB(pat::Muon::PVDZ));
	  C_mu2_Phi_BS_ip_xy  ->push_back(iMuonPhi2->dB(pat::Muon::BS2D));
	  C_mu2_Phi_BS_ip     ->push_back(iMuonPhi2->dB(pat::Muon::BS3D));
	  C_mu2_Phi_PV_ip_z   ->push_back(iMuonPhi2->dB(pat::Muon::PVDZ));
	  C_mu2_Phi_charge->push_back(iMuonPhi2->charge());
	  C_mu2_Phi_isSoft   ->push_back(isSoftMuonPhi2? 1: 0);
	  C_mu2_Phi_isLoose  ->push_back(isLooseMuonPhi2? 1: 0);
	  C_mu2_Phi_isMedium ->push_back(isMediumMuonPhi2? 1: 0);
	  C_mu2_Phi_idx ->push_back(i_muphi2);

	  C_mu1_Phi_px ->push_back(iMuonPhi1->px());
	  C_mu1_Phi_py ->push_back(iMuonPhi1->py());
	  C_mu1_Phi_pz ->push_back(iMuonPhi1->pz());
	  C_mu1_Phi_pt ->push_back(iMuonPhi1->pt());
	  C_mu1_Phi_eta ->push_back(iMuonPhi1->eta());
	  C_mu1_Phi_BS_ips_xy ->push_back(iMuonPhi1->dB(pat::Muon::BS2D)/iMuonPhi1->edB(pat::Muon::BS2D));
	  C_mu1_Phi_BS_ips    ->push_back(iMuonPhi1->dB(pat::Muon::BS3D)/iMuonPhi1->edB(pat::Muon::BS3D));
	  C_mu1_Phi_PV_ips_z  ->push_back(iMuonPhi1->dB(pat::Muon::PVDZ)/iMuonPhi1->edB(pat::Muon::PVDZ));
	  C_mu1_Phi_BS_ip_xy  ->push_back(iMuonPhi1->dB(pat::Muon::BS2D));
	  C_mu1_Phi_BS_ip     ->push_back(iMuonPhi1->dB(pat::Muon::BS3D));
	  C_mu1_Phi_PV_ip_z   ->push_back(iMuonPhi1->dB(pat::Muon::PVDZ));
	  C_mu1_Phi_charge ->push_back(iMuonPhi1->charge());
	  C_mu1_Phi_isSoft   ->push_back(isSoftMuonPhi1? 1: 0);
	  C_mu1_Phi_isLoose  ->push_back(isLooseMuonPhi1? 1: 0);
	  C_mu1_Phi_isMedium ->push_back(isMediumMuonPhi1? 1: 0);
	  C_mu1_Phi_idx ->push_back(i_muphi1);

	  C_pi_charge ->push_back(iTrack1->charge());
	  C_pi_px ->push_back(p4pi1.Px());
	  C_pi_py ->push_back(p4pi1.Py());
	  C_pi_pz ->push_back(p4pi1.Pz());
	  C_pi_pt ->push_back(p4pi1.Pt());
	  C_pi_eta ->push_back(p4pi1.Eta());
	  C_pi_BS_ips_xy->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position()))/std::abs(iTrack1->dxyError()));
	  C_pi_BS_ips_z ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position()))/std::abs(iTrack1->dzError()));
	  C_pi_BS_ip_xy ->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position())));
	  C_pi_BS_ip_z  ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position())));
	}
      }
    }


  // ===================== END OF EVENT : WRITE ETC ++++++++++++++++++++++

  if (nCand > 0)
  {
    cout << "_____________________ SUCCESS!!!! _______________________" << endl;
    N_written_events++;
    cout << N_written_events << " candidates are written to the file now " << endl;
    cout << endl;

    wwtree->Fill();
  }

  C_Ds_vertex_prob->clear();
  C_Ds_mass->clear(); 
  C_Ds_preFit_mass->clear(); 
  C_Ds_px->clear();
  C_Ds_py->clear();
  C_Ds_pz->clear();
  C_Ds_pt->clear();
  C_Ds_vertex_x->clear();
  C_Ds_vertex_y->clear();
  C_Ds_vertex_z->clear();
  C_Ds_vertex_xErr->clear();
  C_Ds_vertex_yErr->clear();
  C_Ds_vertex_zErr->clear();
  C_Ds_vertex_sig->clear();
  C_Ds_vertex_cos3D->clear();
  C_Ds_vertex_cos2D->clear();

  C_Phi_vertex_prob->clear();
  C_Phi_mass->clear(); 
  C_Phi_preFit_mass->clear(); 
  C_Phi_px->clear();
  C_Phi_py->clear();
  C_Phi_pz->clear();
  //
  C_mu2_Phi_px->clear();
  C_mu2_Phi_py->clear();
  C_mu2_Phi_pz->clear();
  C_mu2_Phi_pt->clear();
  C_mu2_Phi_eta->clear();
  C_mu2_Phi_BS_ips_xy->clear();
  C_mu2_Phi_BS_ips->clear();
  C_mu2_Phi_PV_ips_z->clear();
  C_mu2_Phi_BS_ip_xy->clear();
  C_mu2_Phi_BS_ip->clear();
  C_mu2_Phi_PV_ip_z->clear();
  C_mu2_Phi_charge->clear();
  C_mu2_Phi_isSoft->clear();
  C_mu2_Phi_isLoose->clear();
  C_mu2_Phi_isMedium->clear();
  C_mu2_Phi_idx->clear();

  C_mu1_Phi_px->clear();
  C_mu1_Phi_py->clear();
  C_mu1_Phi_pz->clear();
  C_mu1_Phi_pt->clear();
  C_mu1_Phi_eta->clear();
  C_mu1_Phi_BS_ips_xy->clear();
  C_mu1_Phi_BS_ips->clear();
  C_mu1_Phi_PV_ips_z->clear();
  C_mu1_Phi_BS_ip_xy->clear();
  C_mu1_Phi_BS_ip->clear();
  C_mu1_Phi_PV_ip_z->clear();
  C_mu1_Phi_charge->clear();
  C_mu1_Phi_isSoft->clear();
  C_mu1_Phi_isLoose->clear();
  C_mu1_Phi_isMedium->clear();
  C_mu1_Phi_idx->clear();

  C_mu_trig_idx->clear();
  C_mu_trig_pt->clear();
  C_mu_trig_eta->clear();
  C_mu_trig_phi->clear();

  C_pi_charge->clear();
  C_pi_px->clear(); 
  C_pi_py->clear();
  C_pi_pz->clear();
  C_pi_pt->clear();
  C_pi_eta->clear();
  C_pi_BS_ips_xy->clear();
  C_pi_BS_ips_z->clear();
  C_pi_BS_ip_xy->clear();
  C_pi_BS_ip_z->clear();

  PV_x->clear();   PV_y->clear();   PV_z->clear();
  PV_xErr->clear();   PV_yErr->clear();   PV_zErr->clear();
  PV_prob->clear();   //PV_dN->clear();

  HLT_mu7_ip4_matched->clear();
  HLT_mu8_ip3_matched->clear();
  HLT_mu8_ip3p5_matched ->clear();
  HLT_mu8_ip5_matched->clear();
  HLT_mu8_ip6_matched->clear();
  HLT_mu9_ip4_matched->clear();
  HLT_mu9_ip5_matched->clear();
  HLT_mu9_ip6_matched->clear();
  HLT_mu10p5_ip3p5_matched->clear();
  HLT_mu12_ip6_matched->clear();

  HLT_mu7_ip4_eta->clear();
  HLT_mu8_ip3_eta->clear();
  HLT_mu8_ip3p5_eta ->clear();
  HLT_mu8_ip5_eta->clear();
  HLT_mu8_ip6_eta->clear();
  HLT_mu9_ip4_eta->clear();
  HLT_mu9_ip5_eta->clear();
  HLT_mu9_ip6_eta->clear();
  HLT_mu10p5_ip3p5_eta->clear();
  HLT_mu12_ip6_eta->clear();

  HLT_mu7_ip4_pt->clear();
  HLT_mu8_ip3_pt->clear();
  HLT_mu8_ip3p5_pt ->clear();
  HLT_mu8_ip5_pt->clear();
  HLT_mu8_ip6_pt->clear();
  HLT_mu9_ip4_pt->clear();
  HLT_mu9_ip5_pt->clear();
  HLT_mu9_ip6_pt->clear();
  HLT_mu10p5_ip3p5_pt->clear();
  HLT_mu12_ip6_pt->clear();

  HLT_mu7_ip4_dr->clear();
  HLT_mu8_ip3_dr->clear();
  HLT_mu8_ip3p5_dr ->clear();
  HLT_mu8_ip5_dr->clear();
  HLT_mu8_ip6_dr->clear();
  HLT_mu9_ip4_dr->clear();
  HLT_mu9_ip5_dr->clear();
  HLT_mu9_ip6_dr->clear();
  HLT_mu10p5_ip3p5_dr->clear();
  HLT_mu12_ip6_dr->clear();

  HLT_mu7_ip4_fired->clear();
  HLT_mu8_ip3_fired->clear();
  HLT_mu8_ip3p5_fired ->clear();
  HLT_mu8_ip5_fired->clear();
  HLT_mu8_ip6_fired->clear();
  HLT_mu9_ip4_fired->clear();
  HLT_mu9_ip5_fired->clear();
  HLT_mu9_ip6_fired->clear();
  HLT_mu10p5_ip3p5_fired->clear();
  HLT_mu12_ip6_fired->clear();


}

template <typename A,typename B> bool DsToPhiPi_PhiToMuMu_prompt_miniAOD::IsTheSame(const A& cand1, const B& cand2){
  double deltaPt  = std::abs(cand1.pt()-cand2.pt());
  double deltaEta = cand1.eta()-cand2.eta();

  auto deltaPhi = std::abs(cand1.phi() - cand2.phi());
  if (deltaPhi > float(M_PI))
    deltaPhi -= float(2 * M_PI);
  double deltaR2 = deltaEta*deltaEta + deltaPhi*deltaPhi;
  if(deltaR2<0.01 && deltaPt<0.1) return true;
  else return false;
}

int DsToPhiPi_PhiToMuMu_prompt_miniAOD::getMatchedGenPartIdx(double pt, double eta, double phi, int pdg_id, std::vector<pat::PackedGenParticle> packedGen_particles){
  int matchedIndex = -9999;
  float max_dr2 = 9999.;

  for(unsigned i=0; i<packedGen_particles.size(); ++i){
    pat::PackedGenParticle gen_particle = packedGen_particles.at(i);
    if (std::abs(gen_particle.pdgId()) != pdg_id) continue;
    double gen_pt  = gen_particle.pt();
    double gen_eta = gen_particle.eta();
    double gen_phi = gen_particle.phi();

    double deltaPt  = std::abs(pt-gen_pt);
    double deltaEta = eta-gen_eta;

    auto deltaPhi = std::abs(phi - gen_phi);
    if (deltaPhi > float(M_PI))
      deltaPhi -= float(2 * M_PI);

    double deltaR2 = deltaEta*deltaEta + deltaPhi*deltaPhi;
    if (deltaPt<0.5 && deltaR2<0.25 && deltaR2<max_dr2) {
      matchedIndex = i;
      max_dr2 = deltaR2;
    }
  }

  return matchedIndex;  
}


// ------------ method called once each job just before starting event loop  ------------
void DsToPhiPi_PhiToMuMu_prompt_miniAOD::beginJob()
{
  cout << "------------------------------->>>>> Begin Job" << endl;

  f = new TFile(fileName.c_str(), "RECREATE");
  wwtree  = new TTree("wztree", "muons tree");

  wwtree->Branch("nCand"              , &nCand            , "nCand/I"     );

  wwtree->Branch("run"                , &run              , "run/I"       );
  wwtree->Branch("event"              , &event            , "event/I"     );
  wwtree->Branch("lumi"               , &lumi             , "lumi/F"      );

  wwtree->Branch("nPV"              , &nPV            , "nPV/I"     );
  wwtree->Branch("nPU_trueInt"      , &nPU_trueInt    , "nPU_trueInt/I");
  wwtree->Branch("numTrack"           , &numTrack         , "numTrack/I"  );

  wwtree->Branch("C_Ds_vertex_prob", &C_Ds_vertex_prob);
  wwtree->Branch("C_Ds_mass", &C_Ds_mass);
  wwtree->Branch("C_Ds_preFit_mass", &C_Ds_preFit_mass);
  wwtree->Branch("C_Ds_px", &C_Ds_px);
  wwtree->Branch("C_Ds_py", &C_Ds_py);
  wwtree->Branch("C_Ds_pz", &C_Ds_pz);
  wwtree->Branch("C_Ds_pt", &C_Ds_pt);
  wwtree->Branch("C_Ds_vertex_x" , &C_Ds_vertex_x);
  wwtree->Branch("C_Ds_vertex_y" , &C_Ds_vertex_y);
  wwtree->Branch("C_Ds_vertex_z" , &C_Ds_vertex_z);
  wwtree->Branch("C_Ds_vertex_xErr" , &C_Ds_vertex_xErr);
  wwtree->Branch("C_Ds_vertex_yErr" , &C_Ds_vertex_yErr);
  wwtree->Branch("C_Ds_vertex_zErr" , &C_Ds_vertex_zErr);
  wwtree->Branch("C_Ds_vertex_sig" , &C_Ds_vertex_sig);
  wwtree->Branch("C_Ds_vertex_cos3D" , &C_Ds_vertex_cos3D);
  wwtree->Branch("C_Ds_vertex_cos2D" , &C_Ds_vertex_cos2D);

  wwtree->Branch("C_Phi_vertex_prob", &C_Phi_vertex_prob);
  wwtree->Branch("C_Phi_mass", &C_Phi_mass);
  wwtree->Branch("C_Phi_preFit_mass", &C_Phi_preFit_mass);
  wwtree->Branch("C_Phi_px", &C_Phi_px);
  wwtree->Branch("C_Phi_py", &C_Phi_py);
  wwtree->Branch("C_Phi_pz", &C_Phi_pz);

  wwtree->Branch("C_mu2_Phi_px"    , &C_mu2_Phi_px);
  wwtree->Branch("C_mu2_Phi_py"    , &C_mu2_Phi_py);
  wwtree->Branch("C_mu2_Phi_pz"    , &C_mu2_Phi_pz);
  wwtree->Branch("C_mu2_Phi_pt"   , &C_mu2_Phi_pt);
  wwtree->Branch("C_mu2_Phi_eta"   , &C_mu2_Phi_eta);
  wwtree->Branch("C_mu2_Phi_BS_ips_xy", &C_mu2_Phi_BS_ips_xy);
  wwtree->Branch("C_mu2_Phi_BS_ips", &C_mu2_Phi_BS_ips);
  wwtree->Branch("C_mu2_Phi_PV_ips_z" , &C_mu2_Phi_PV_ips_z);
  wwtree->Branch("C_mu2_Phi_BS_ip_xy" , &C_mu2_Phi_BS_ip_xy);
  wwtree->Branch("C_mu2_Phi_BS_ip" , &C_mu2_Phi_BS_ip);
  wwtree->Branch("C_mu2_Phi_PV_ip_z"  , &C_mu2_Phi_PV_ip_z);
  wwtree->Branch("C_mu2_Phi_charge", &C_mu2_Phi_charge);
  wwtree->Branch("C_mu2_Phi_isSoft", &C_mu2_Phi_isSoft);
  wwtree->Branch("C_mu2_Phi_isLoose", &C_mu2_Phi_isLoose);
  wwtree->Branch("C_mu2_Phi_isMedium", &C_mu2_Phi_isMedium);
  wwtree->Branch("C_mu2_Phi_idx", &C_mu2_Phi_idx);

  wwtree->Branch("C_mu1_Phi_px"    , &C_mu1_Phi_px);
  wwtree->Branch("C_mu1_Phi_py"    , &C_mu1_Phi_py);
  wwtree->Branch("C_mu1_Phi_pz"    , &C_mu1_Phi_pz);
  wwtree->Branch("C_mu1_Phi_pt"   , &C_mu1_Phi_pt);
  wwtree->Branch("C_mu1_Phi_eta"   , &C_mu1_Phi_eta);
  wwtree->Branch("C_mu1_Phi_BS_ips_xy", &C_mu1_Phi_BS_ips_xy);
  wwtree->Branch("C_mu1_Phi_BS_ips", &C_mu1_Phi_BS_ips);
  wwtree->Branch("C_mu1_Phi_PV_ips_z" , &C_mu1_Phi_PV_ips_z);
  wwtree->Branch("C_mu1_Phi_BS_ip_xy" , &C_mu1_Phi_BS_ip_xy);
  wwtree->Branch("C_mu1_Phi_BS_ip" , &C_mu1_Phi_BS_ip);
  wwtree->Branch("C_mu1_Phi_PV_ip_z"  , &C_mu1_Phi_PV_ip_z);
  wwtree->Branch("C_mu1_Phi_charge", &C_mu1_Phi_charge);
  wwtree->Branch("C_mu1_Phi_isSoft", &C_mu1_Phi_isSoft);
  wwtree->Branch("C_mu1_Phi_isLoose", &C_mu1_Phi_isLoose);
  wwtree->Branch("C_mu1_Phi_isMedium", &C_mu1_Phi_isMedium);
  wwtree->Branch("C_mu1_Phi_idx", &C_mu1_Phi_idx);

  wwtree->Branch("C_mu_trig_idx", &C_mu_trig_idx);
  wwtree->Branch("C_mu_trig_pt",  &C_mu_trig_pt);
  wwtree->Branch("C_mu_trig_eta", &C_mu_trig_eta);
  wwtree->Branch("C_mu_trig_phi", &C_mu_trig_phi);

  wwtree->Branch("C_pi_charge"    , &C_pi_charge     );
  wwtree->Branch("C_pi_px"        , &C_pi_px         );
  wwtree->Branch("C_pi_py"        , &C_pi_py         );
  wwtree->Branch("C_pi_pz"        , &C_pi_pz         );
  wwtree->Branch("C_pi_pt"        , &C_pi_pt        );
  wwtree->Branch("C_pi_eta"       , &C_pi_eta        );
  wwtree->Branch("C_pi_BS_ips_xy" , &C_pi_BS_ips_xy     );
  wwtree->Branch("C_pi_BS_ips_z"  , &C_pi_BS_ips_z     );
  wwtree->Branch("C_pi_BS_ip_xy"  , &C_pi_BS_ip_xy     );
  wwtree->Branch("C_pi_BS_ip_z"   , &C_pi_BS_ip_z     );

  wwtree->Branch("PV_x"    , &PV_x);
  wwtree->Branch("PV_y"    , &PV_y);
  wwtree->Branch("PV_z"    , &PV_z);
  wwtree->Branch("PV_xErr" , &PV_xErr);
  wwtree->Branch("PV_yErr" , &PV_yErr);
  wwtree->Branch("PV_zErr" , &PV_zErr);
  wwtree->Branch("PV_prob" , &PV_prob);
  //wwtree->Branch("PV_dN"   , &PV_dN);

  wwtree->Branch("HLT_mu7_ip4_matched",    &HLT_mu7_ip4_matched);
  wwtree->Branch("HLT_mu8_ip3_matched",    &HLT_mu8_ip3_matched);
  wwtree->Branch("HLT_mu8_ip3p5_matched",  &HLT_mu8_ip3p5_matched);
  wwtree->Branch("HLT_mu8_ip5_matched",    &HLT_mu8_ip5_matched);
  wwtree->Branch("HLT_mu8_ip6_matched",    &HLT_mu8_ip6_matched);
  wwtree->Branch("HLT_mu9_ip4_matched",    &HLT_mu9_ip4_matched);
  wwtree->Branch("HLT_mu9_ip5_matched",    &HLT_mu9_ip5_matched);
  wwtree->Branch("HLT_mu9_ip6_matched",    &HLT_mu9_ip6_matched);
  wwtree->Branch("HLT_mu10p5_ip3p5_matched", &HLT_mu10p5_ip3p5_matched);
  wwtree->Branch("HLT_mu12_ip6_matched",   &HLT_mu12_ip6_matched);

  wwtree->Branch("HLT_mu7_ip4_eta",    &HLT_mu7_ip4_eta);
  wwtree->Branch("HLT_mu8_ip3_eta",    &HLT_mu8_ip3_eta);
  wwtree->Branch("HLT_mu8_ip3p5_eta",  &HLT_mu8_ip3p5_eta);
  wwtree->Branch("HLT_mu8_ip5_eta",    &HLT_mu8_ip5_eta);
  wwtree->Branch("HLT_mu8_ip6_eta",    &HLT_mu8_ip6_eta);
  wwtree->Branch("HLT_mu9_ip4_eta",    &HLT_mu9_ip4_eta);
  wwtree->Branch("HLT_mu9_ip5_eta",    &HLT_mu9_ip5_eta);
  wwtree->Branch("HLT_mu9_ip6_eta",    &HLT_mu9_ip6_eta);
  wwtree->Branch("HLT_mu10p5_ip3p5_eta", &HLT_mu10p5_ip3p5_eta);
  wwtree->Branch("HLT_mu12_ip6_eta",   &HLT_mu12_ip6_eta);

  wwtree->Branch("HLT_mu7_ip4_pt",    &HLT_mu7_ip4_pt);
  wwtree->Branch("HLT_mu8_ip3_pt",    &HLT_mu8_ip3_pt);
  wwtree->Branch("HLT_mu8_ip3p5_pt",  &HLT_mu8_ip3p5_pt);
  wwtree->Branch("HLT_mu8_ip5_pt",    &HLT_mu8_ip5_pt);
  wwtree->Branch("HLT_mu8_ip6_pt",    &HLT_mu8_ip6_pt);
  wwtree->Branch("HLT_mu9_ip4_pt",    &HLT_mu9_ip4_pt);
  wwtree->Branch("HLT_mu9_ip5_pt",    &HLT_mu9_ip5_pt);
  wwtree->Branch("HLT_mu9_ip6_pt",    &HLT_mu9_ip6_pt);
  wwtree->Branch("HLT_mu10p5_ip3p5_pt", &HLT_mu10p5_ip3p5_pt);
  wwtree->Branch("HLT_mu12_ip6_pt",   &HLT_mu12_ip6_pt);

  wwtree->Branch("HLT_mu7_ip4_dr",    &HLT_mu7_ip4_dr);
  wwtree->Branch("HLT_mu8_ip3_dr",    &HLT_mu8_ip3_dr);
  wwtree->Branch("HLT_mu8_ip3p5_dr",  &HLT_mu8_ip3p5_dr);
  wwtree->Branch("HLT_mu8_ip5_dr",    &HLT_mu8_ip5_dr);
  wwtree->Branch("HLT_mu8_ip6_dr",    &HLT_mu8_ip6_dr);
  wwtree->Branch("HLT_mu9_ip4_dr",    &HLT_mu9_ip4_dr);
  wwtree->Branch("HLT_mu9_ip5_dr",    &HLT_mu9_ip5_dr);
  wwtree->Branch("HLT_mu9_ip6_dr",    &HLT_mu9_ip6_dr);
  wwtree->Branch("HLT_mu10p5_ip3p5_dr", &HLT_mu10p5_ip3p5_dr);
  wwtree->Branch("HLT_mu12_ip6_dr",   &HLT_mu12_ip6_dr);

  wwtree->Branch("HLT_mu7_ip4_fired",    &HLT_mu7_ip4_fired);
  wwtree->Branch("HLT_mu8_ip3_fired",    &HLT_mu8_ip3_fired);
  wwtree->Branch("HLT_mu8_ip3p5_fired",  &HLT_mu8_ip3p5_fired);
  wwtree->Branch("HLT_mu8_ip5_fired",    &HLT_mu8_ip5_fired);
  wwtree->Branch("HLT_mu8_ip6_fired",    &HLT_mu8_ip6_fired);
  wwtree->Branch("HLT_mu9_ip4_fired",    &HLT_mu9_ip4_fired);
  wwtree->Branch("HLT_mu9_ip5_fired",    &HLT_mu9_ip5_fired);
  wwtree->Branch("HLT_mu9_ip6_fired",    &HLT_mu9_ip6_fired);
  wwtree->Branch("HLT_mu10p5_ip3p5_fired", &HLT_mu10p5_ip3p5_fired);
  wwtree->Branch("HLT_mu12_ip6_fired",   &HLT_mu12_ip6_fired);


}

// ------------ method called once each job just after ending the event loop  ------------
  void
DsToPhiPi_PhiToMuMu_prompt_miniAOD::endJob()
{
  cout << "------------------------------->>>>> Ending Job" << endl;
  cout << endl;
  cout <<  "total " << N_written_events << " candidates were written to the file" << endl;
  cout << "------------------------------->>>>> End Job" << endl;

  f->WriteTObject(wwtree);
  delete wwtree;
  f->Close();

}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DsToPhiPi_PhiToMuMu_prompt_miniAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DsToPhiPi_PhiToMuMu_prompt_miniAOD);
