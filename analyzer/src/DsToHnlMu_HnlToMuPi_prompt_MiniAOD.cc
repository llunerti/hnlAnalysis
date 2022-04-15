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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

//// gen ??
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"

#include "DataFormats/Math/interface/deltaR.h"

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
class DsToHnlMu_HnlToMuPi_prompt_miniAOD : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
  public:
    explicit DsToHnlMu_HnlToMuPi_prompt_miniAOD(const edm::ParameterSet&);
    ~DsToHnlMu_HnlToMuPi_prompt_miniAOD();

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
    double trig_mu_pt_cut_;
    double trig_mu_eta_cut_;
    double pi_pt_cut_;
    double pi_eta_cut_;
    double mumupi_mass_cut_;
    double mupi_mass_high_cut_;
    double mupi_mass_low_cut_;
    double mupi_pt_cut_;
    double vtx_prob_cut_;
    int isSignal;

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

    std::vector<float> *C_Hnl_vertex_prob;
    std::vector<float> *C_Hnl_mass;
    std::vector<float> *C_Hnl_preFit_mass;
    std::vector<float> *C_Hnl_px, *C_Hnl_py, *C_Hnl_pz;
    std::vector<float> *C_Hnl_pt;
    std::vector<float> *C_Hnl_vertex_x, *C_Hnl_vertex_y, *C_Hnl_vertex_z;
    std::vector<float> *C_Hnl_vertex_xErr, *C_Hnl_vertex_yErr, *C_Hnl_vertex_zErr;
    std::vector<float> *C_Hnl_vertex_sig;
    std::vector<float> *C_Hnl_vertex_cos3D;
    std::vector<float> *C_Hnl_vertex_cos2D;

    std::vector<float> *C_mu_Hnl_px, *C_mu_Hnl_py, *C_mu_Hnl_pz;
    std::vector<float> *C_mu_Hnl_pt;
    std::vector<float> *C_mu_Hnl_eta;
    std::vector<float> *C_mu_Hnl_phi;
    std::vector<float> *C_mu_Hnl_BS_ips_xy, *C_mu_Hnl_PV_ips_z;
    std::vector<float> *C_mu_Hnl_BS_ips;
    std::vector<float> *C_mu_Hnl_BS_ip_xy, *C_mu_Hnl_PV_ip_z;
    std::vector<float> *C_mu_Hnl_BS_ip;
    std::vector<int>   *C_mu_Hnl_charge;
    std::vector<short> *C_mu_Hnl_isSoft;
    std::vector<short> *C_mu_Hnl_isLoose;
    std::vector<short> *C_mu_Hnl_isMedium;
    std::vector<short> *C_mu_Hnl_isGlobal;
    std::vector<short> *C_mu_Hnl_isTracker;
    std::vector<short> *C_mu_Hnl_isStandAlone;
    std::vector<short> *C_mu_Hnl_isMCMatched;
    std::vector<unsigned> *C_mu_Hnl_idx;
    std::vector<int>   *C_mu_Hnl_idMatch;

    std::vector<float> *C_mu_Ds_px, *C_mu_Ds_py, *C_mu_Ds_pz;
    std::vector<float> *C_mu_Ds_pt;
    std::vector<float> *C_mu_Ds_eta;
    std::vector<float> *C_mu_Ds_phi;
    std::vector<float> *C_mu_Ds_BS_ips_xy, *C_mu_Ds_PV_ips_z;
    std::vector<float> *C_mu_Ds_BS_ips;
    std::vector<float> *C_mu_Ds_BS_ip_xy, *C_mu_Ds_PV_ip_z;
    std::vector<float> *C_mu_Ds_BS_ip;
    std::vector<int>   *C_mu_Ds_charge;
    std::vector<short> *C_mu_Ds_isSoft;
    std::vector<short> *C_mu_Ds_isLoose;
    std::vector<short> *C_mu_Ds_isMedium;
    std::vector<short> *C_mu_Ds_isGlobal;
    std::vector<short> *C_mu_Ds_isTracker;
    std::vector<short> *C_mu_Ds_isStandAlone;
    std::vector<short> *C_mu_Ds_isMCMatched;
    std::vector<unsigned> *C_mu_Ds_idx;

    std::vector<float> *C_mu1mu2_mass;
    std::vector<float> *C_mu1mu2_dr;
    std::vector<float> *C_mu1pi_dr;
    std::vector<float> *C_mu2pi_dr;

    std::vector<int>   *C_pi_charge;
    std::vector<float> *C_pi_px, *C_pi_py, *C_pi_pz;
    std::vector<float> *C_pi_pt;
    std::vector<float> *C_pi_eta;
    std::vector<float> *C_pi_phi;
    std::vector<float> *C_pi_BS_ips_xy, *C_pi_BS_ips_z;
    std::vector<float> *C_pi_BS_ip_xy, *C_pi_BS_ip_z;
    std::vector<short> *C_pi_isMCMatched;

    std::vector<float> *PV_x, *PV_y, *PV_z;
    std::vector<float> *PV_xErr, *PV_yErr, *PV_zErr;
    std::vector<float> *PV_prob;
    //std::vector<int>   *PV_dN;

    std::vector<unsigned>  *HLT_mu_trig_idx;
    std::vector<float>  *HLT_mu_trig_pt;
    std::vector<float>  *HLT_mu_trig_eta;

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


    std::vector<short>  *C_pi_isHnlDaughter;
    std::vector<short>  *C_mu_Hnl_isHnlDaughter;
    std::vector<short>  *C_mu_Ds_isHnlBrother;

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


DsToHnlMu_HnlToMuPi_prompt_miniAOD::DsToHnlMu_HnlToMuPi_prompt_miniAOD(const edm::ParameterSet& iConfig) :
  hlTriggerResults_  (consumes<edm::TriggerResults> (iConfig.getParameter<edm::InputTag>("HLTriggerResults"))),
  hlTriggerObjects_  (consumes<pat::TriggerObjectStandAloneCollection> (iConfig.getParameter<edm::InputTag>("HLTriggerObjects"))),
  prunedGenToken_    (consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("prunedGenParticleTag"))),
  packedGenToken_    (consumes<std::vector<pat::PackedGenParticle>>(iConfig.getParameter<edm::InputTag>("packedGenParticleTag"))),
  beamSpotToken_     (consumes<reco::BeamSpot> (iConfig.getParameter<edm::InputTag>("beamSpotTag"))),
  vtxSample          (consumes<reco::VertexCollection> (iConfig.getParameter<edm::InputTag>("VtxSample"))),
  tracks_            (consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("Track"))),
  muons_             (consumes<std::vector<pat::Muon>> (iConfig.getParameter<edm::InputTag>("muons"))),
  PUInfoToken_       (consumes<std::vector<PileupSummaryInfo>> (iConfig.getParameter<edm::InputTag>("PUInfoTag"))),
  mu_pt_cut_         (iConfig.getUntrackedParameter<double>("mu_pt_cut" )),
  mu_eta_cut_        (iConfig.getUntrackedParameter<double>("mu_eta_cut")),
  trig_mu_pt_cut_    (iConfig.getUntrackedParameter<double>("trigMu_pt_cut" )),
  trig_mu_eta_cut_   (iConfig.getUntrackedParameter<double>("trigMu_eta_cut")),
  pi_pt_cut_         (iConfig.getUntrackedParameter<double>("pi_pt_cut" )),
  pi_eta_cut_        (iConfig.getUntrackedParameter<double>("pi_eta_cut")),
  mumupi_mass_cut_   (iConfig.getUntrackedParameter<double>("b_mass_cut")),
  mupi_mass_high_cut_(iConfig.getUntrackedParameter<double>("mupi_mass_high_cut")),
  mupi_mass_low_cut_ (iConfig.getUntrackedParameter<double>("mupi_mass_low_cut")),
  mupi_pt_cut_       (iConfig.getUntrackedParameter<double>("mupi_pt_cut")),
  vtx_prob_cut_      (iConfig.getUntrackedParameter<double>("vtx_prob_cut")),
  isSignal           (iConfig.getUntrackedParameter<int>("is_signal")),
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

  C_Hnl_vertex_prob(0),
  C_Hnl_mass(0),
  C_Hnl_preFit_mass(0),
  C_Hnl_px(0),
  C_Hnl_py(0),  
  C_Hnl_pz(0),
  C_Hnl_pt(0),
  C_Hnl_vertex_x(0),  
  C_Hnl_vertex_y(0), 
  C_Hnl_vertex_z(0),
  C_Hnl_vertex_xErr(0),  
  C_Hnl_vertex_yErr(0), 
  C_Hnl_vertex_zErr(0),
  C_Hnl_vertex_sig(0),
  C_Hnl_vertex_cos3D(0),
  C_Hnl_vertex_cos2D(0),

  C_mu_Hnl_px(0),
  C_mu_Hnl_py(0),       
  C_mu_Hnl_pz(0),
  C_mu_Hnl_pt(0),
  C_mu_Hnl_eta(0),
  C_mu_Hnl_phi(0),
  C_mu_Hnl_BS_ips_xy(0),
  C_mu_Hnl_PV_ips_z(0),
  C_mu_Hnl_BS_ips(0),
  C_mu_Hnl_BS_ip_xy(0),
  C_mu_Hnl_PV_ip_z(0),
  C_mu_Hnl_BS_ip(0),
  C_mu_Hnl_charge(0),
  C_mu_Hnl_isSoft(0),
  C_mu_Hnl_isLoose(0),
  C_mu_Hnl_isMedium(0),
  C_mu_Hnl_isGlobal(0),
  C_mu_Hnl_isTracker(0),
  C_mu_Hnl_isStandAlone(0),
  C_mu_Hnl_isMCMatched(0),
  C_mu_Hnl_idx(0),

  C_mu_Ds_px(0),
  C_mu_Ds_py(0),
  C_mu_Ds_pz(0),
  C_mu_Ds_pt(0),
  C_mu_Ds_eta(0),
  C_mu_Ds_phi(0),
  C_mu_Ds_BS_ips_xy(0),
  C_mu_Ds_PV_ips_z(0),
  C_mu_Ds_BS_ips(0),
  C_mu_Ds_BS_ip_xy(0),
  C_mu_Ds_PV_ip_z(0),
  C_mu_Ds_BS_ip(0),
  C_mu_Ds_charge(0),
  C_mu_Ds_isSoft(0),
  C_mu_Ds_isLoose(0),
  C_mu_Ds_isMedium(0),
  C_mu_Ds_isGlobal(0),
  C_mu_Ds_isTracker(0),
  C_mu_Ds_isStandAlone(0),
  C_mu_Ds_isMCMatched(0),
  C_mu_Ds_idx(0),

  C_mu1mu2_mass(0),
  C_mu1mu2_dr(0),
  C_mu1pi_dr(0),
  C_mu2pi_dr(0),

  C_pi_charge(0),
  C_pi_px(0),
  C_pi_py(0),
  C_pi_pz(0),
  C_pi_pt(0),
  C_pi_eta(0),
  C_pi_phi(0),
  C_pi_BS_ips_xy(0),
  C_pi_BS_ips_z(0),
  C_pi_BS_ip_xy(0),
  C_pi_BS_ip_z(0),
  C_pi_isMCMatched(0),

  PV_x(0)  , PV_y(0), PV_z(0),
  PV_xErr(0)  , PV_yErr(0), PV_zErr(0),
  PV_prob(0)  , //PV_dN(0),

  HLT_mu_trig_idx(0),
  HLT_mu_trig_pt(0),
  HLT_mu_trig_eta(0),

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

  C_pi_isHnlDaughter(0),
  C_mu_Hnl_isHnlDaughter(0),
  C_mu_Ds_isHnlBrother(0),

  nCand(0),

  run(0),
  event(0),
  lumi(0),

  nPV(0),
  nPU_trueInt(0),
  numTrack(0)
{
  fileName = iConfig.getUntrackedParameter<std::string>("fileName","DsToHnlMu_HnlToMuPi_prompt_2018BPark-MiniAOD.root");
  usesResource("TFileService");
  N_written_events = 0;
  pdg = my_pdg();

}


DsToHnlMu_HnlToMuPi_prompt_miniAOD::~DsToHnlMu_HnlToMuPi_prompt_miniAOD()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
  void
DsToHnlMu_HnlToMuPi_prompt_miniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  //edm::Handle<std::vector<pat::PackedCandidate> > thePATLostTracksHandle;
  //iEvent.getByToken(lostTracks_, thePATLostTracksHandle);

  edm::Handle<std::vector<PileupSummaryInfo>>  PupInfoHandle;
  //iEvent.getByToken(PUInfoToken_, PupInfoHandle);

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

  //trigger muon
  for ( unsigned i_trigmu=0; i_trigmu<thePATMuonHandle->size(); ++i_trigmu){

    const pat::Muon* iTrigMuon = &(*thePATMuonHandle).at(i_trigmu);

    //cut on muon pt and eta
    if(iTrigMuon->pt() < trig_mu_pt_cut_) continue;
    if(std::abs(iTrigMuon->eta()) > trig_mu_eta_cut_)  continue;

    //cut on muon id
    if (!iTrigMuon->isSoftMuon(thePrimaryV)) continue;

    for (unsigned i = 0; i < nTrigPaths; ++i) {

      bool match = false;
      float best_matching_path_dr  = -9999.;
      float best_matching_path_pt  = -9999.;
      float best_matching_path_eta = -9999.;
      float min_dr = 9999.;

      if(iTrigMuon->triggerObjectMatches().size()!=0){

	//loop over trigger object matched to muon
	for(size_t j=0; j<iTrigMuon->triggerObjectMatches().size();j++){

	  if(iTrigMuon->triggerObjectMatch(j)!=0 && iTrigMuon->triggerObjectMatch(j)->hasPathName(TriggerPaths[i],true,true)){

	    float trig_dr  = reco::deltaR(iTrigMuon->triggerObjectMatch(j)->p4(), iTrigMuon->p4()); 
	    float trig_pt  = iTrigMuon->triggerObjectMatch(j)->pt();                   
	    float trig_eta = iTrigMuon->triggerObjectMatch(j)->eta();                   

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

    HLT_mu_trig_idx ->push_back(i_trigmu);
    HLT_mu_trig_pt ->push_back(iTrigMuon->pt());
    HLT_mu_trig_eta ->push_back(iTrigMuon->eta());

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


  for ( unsigned i_muhnl=0; i_muhnl<thePATMuonHandle->size(); ++i_muhnl){

    const pat::Muon* iMuonHnl = &(*thePATMuonHandle).at(i_muhnl);

    //cuts on muon pt and eta
    if (iMuonHnl->pt() < mu_pt_cut_) continue;
    if (std::abs(iMuonHnl->eta()) > mu_eta_cut_)  continue;

    //save muon id info
    bool isSoftMuonHnl = false;
    bool isLooseMuonHnl = false;
    bool isMediumMuonHnl = false;

    if (iMuonHnl->isSoftMuon(thePrimaryV)) isSoftMuonHnl=true;
    if (iMuonHnl->isLooseMuon())           isLooseMuonHnl=true;
    if (iMuonHnl->isMediumMuon())          isMediumMuonHnl=true;

    if (!isSoftMuonHnl && !isLooseMuonHnl && !isMediumMuonHnl) continue;

    //save muon track info
    bool isGlobalMuonHnl = false;
    bool isTrackerMuonHnl = false;
    bool isStandAloneMuonHnl = false;

    if (iMuonHnl->isGlobalMuon())     isGlobalMuonHnl = true;
    if (iMuonHnl->isTrackerMuon())    isTrackerMuonHnl = true;
    if (iMuonHnl->isStandAloneMuon()) isStandAloneMuonHnl = true;

    //cuts on muon track
    TrackRef inTrackMuHnl;
    inTrackMuHnl = iMuonHnl->track();
    if( inTrackMuHnl.isNull())  continue;
    if(!(inTrackMuHnl->quality(reco::TrackBase::highPurity)))  continue;


    for (std::vector<pat::PackedCandidate>::const_iterator iTrack1 = thePATTrackHandle->begin(); iTrack1 != thePATTrackHandle->end(); ++iTrack1){

      if (IsTheSame(*iTrack1,*iMuonHnl) ) continue;

      //Nota bene: if you want to use dxy or dz you need to be sure 
      //the pt of the tracks is bigger than 0.5 GeV, otherwise you 
      //will get an error related to covariance matrix.
      //Next lines are very recommended

      if(iTrack1->pt() <= pi_pt_cut_) continue;
      if(std::abs(iTrack1->eta()) > pi_eta_cut_) continue;
      if(iTrack1->charge()==0) continue;// NO neutral objects
      if(std::abs(iTrack1->pdgId())!=211) continue;//Due to the lack of the particle ID all the tracks for cms are pions(ID==211)
      if(!(iTrack1->trackHighPurity())) continue; 


      bool hnl_C_pi_match = false;
      bool isMCMatchedTrack1 = false;

      if(run==1 && isSignal>0){
	int match_C_pi_idx = getMatchedGenPartIdx(iTrack1->pt(),iTrack1->eta(),iTrack1->phi(),211,*packedGenParticleCollection);

	if (match_C_pi_idx>0){
	  isMCMatchedTrack1 = true;
	  pat::PackedGenParticle matchedGenPi = (*packedGenParticleCollection).at(match_C_pi_idx);
	  if (matchedGenPi.motherRef().isNonnull() &&
	      matchedGenPi.motherRef().isAvailable() &&
	      std::abs(matchedGenPi.mother(0)->pdgId()) == 9900015 ){
	    hnl_C_pi_match=true;
	  }
	}
      }

      //cuts on mupi mass and pt
      TLorentzVector p4muhnl,p4pi;
      p4pi.SetPtEtaPhiM(iTrack1->pt(),iTrack1->eta(),iTrack1->phi(), pdg.PDG_PION_MASS);
      p4muhnl.SetPtEtaPhiM(iMuonHnl->pt(), iMuonHnl->eta(), iMuonHnl->phi(), pdg.PDG_MUON_MASS);

      if ((p4muhnl + p4pi).M() >  mupi_mass_high_cut_) continue;
      if ((p4muhnl + p4pi).M() <  mupi_mass_low_cut_) continue;
      if ((p4muhnl + p4pi).Pt() <  mupi_pt_cut_) continue;


      TransientTrack muonHnlTT((*TTrackBuilder).build(inTrackMuHnl));
      TransientTrack pion1TT((*TTrackBuilder).build(iTrack1->pseudoTrack()));


      if(!muonHnlTT.isValid()) continue;
      if(!pion1TT.isValid()) continue;
      if(muonHnlTT == pion1TT) continue;


      float muon_sigma = pdg.PDG_MUON_MASS * 1.e-6;
      float chi = 0.;
      float ndf = 0.;


      bool hnl_mu_match = false;
      bool isMCMatchedMuonHnl = false;

      if(run==1 && isSignal>0){
	int match_C_mu_Hnl_idx = getMatchedGenPartIdx(iMuonHnl->pt(),iMuonHnl->eta(),iMuonHnl->phi(),13,*packedGenParticleCollection);

	if (match_C_mu_Hnl_idx>0){
	  isMCMatchedMuonHnl = true;
	  pat::PackedGenParticle matchedGenMuon = (*packedGenParticleCollection).at(match_C_mu_Hnl_idx);
	  if (matchedGenMuon.motherRef().isNonnull() &&
	      matchedGenMuon.motherRef().isAvailable() &&
	      std::abs(matchedGenMuon.mother(0)->pdgId()) == 9900015 ){
	    hnl_mu_match=true;
	  }
	}
      }

      //add muon from Ds
      for ( unsigned i_muds=0; i_muds<thePATMuonHandle->size(); ++i_muds){

	if (i_muds==i_muhnl) continue;

	const pat::Muon* iMuonDs = &(*thePATMuonHandle).at(i_muds);

	if (IsTheSame(*iTrack1,*iMuonDs) ) continue;

	//cuts on muon pt and eta
	if (iMuonDs->pt() < mu_pt_cut_) continue;
	if (std::abs(iMuonDs->eta()) > mu_eta_cut_)  continue;

	//save muon id info
	bool isSoftMuonDs = false;
	bool isLooseMuonDs = false;
	bool isMediumMuonDs = false;

	if (iMuonDs->isSoftMuon(thePrimaryV)) isSoftMuonDs=true;
	if (iMuonDs->isLooseMuon())           isLooseMuonDs=true;
	if (iMuonDs->isMediumMuon())          isMediumMuonDs=true;

	//save muon track info
	bool isGlobalMuonDs = false;
	bool isTrackerMuonDs = false;
	bool isStandAloneMuonDs = false;

	if (iMuonDs->isGlobalMuon())     isGlobalMuonDs = true;
	if (iMuonDs->isTrackerMuon())    isTrackerMuonDs = true;
	if (iMuonDs->isStandAloneMuon()) isStandAloneMuonDs = true;

	//cuts on muon track
	TrackRef inTrackMuDs;
	inTrackMuDs = iMuonDs->track();
	if( inTrackMuDs.isNull())  continue;
	if(!(inTrackMuDs->quality(reco::TrackBase::highPurity)))  continue;

	TLorentzVector p4muds;
	p4muds.SetPtEtaPhiM(iMuonDs->pt(), iMuonDs->eta(), iMuonDs->phi(), pdg.PDG_MUON_MASS);

	if ((p4muhnl + p4muds + p4pi).M() > mumupi_mass_cut_) continue;

	TransientTrack muonDsTT((*TTrackBuilder).build(inTrackMuDs));
	if(!muonDsTT.isValid()) continue;

	//Initialize Hnl->MuPi particles
	std::vector<RefCountedKinematicParticle> hnlParticles;
	hnlParticles.push_back(pFactory.particle(muonHnlTT, pdg.PM_PDG_MUON_MASS, chi, ndf, muon_sigma));
	hnlParticles.push_back(pFactory.particle(pion1TT, pdg.PM_PDG_PION_MASS, chi, ndf, muon_sigma));

	//Fit Hnl->MuPi 
	KinematicParticleVertexFitter hnlToPiMu_vertexFitter;
	RefCountedKinematicTree hnlToPiMu_kinTree;
	hnlToPiMu_kinTree = hnlToPiMu_vertexFitter.fit(hnlParticles);

	if (!hnlToPiMu_kinTree->isValid()) continue;
	hnlToPiMu_kinTree->movePointerToTheTop();
	RefCountedKinematicParticle hnl_particle = hnlToPiMu_kinTree->currentParticle();
	RefCountedKinematicVertex   hnl_vtx      = hnlToPiMu_kinTree->currentDecayVertex();

	double hnl_mass = hnl_particle->currentState().mass();


	double hnl_vtxprob = TMath::Prob(hnl_vtx->chiSquared(), hnl_vtx->degreesOfFreedom());
	if(hnl_vtxprob < vtx_prob_cut_) continue;

	//Initialize Ds->HnlMu particles
	std::vector<RefCountedKinematicParticle> dsParticles;
	dsParticles.push_back(pFactory.particle(muonDsTT, pdg.PM_PDG_MUON_MASS, chi, ndf, muon_sigma)); // mu from Ds
	dsParticles.push_back(hnl_particle); //add hnl

	//Fit Ds->HnlMu 
	RefCountedKinematicTree dsToHnlMu_kinTree;
	KinematicParticleVertexFitter dsToHnlMu_vertexFitter;

        //this protects the fit from failing
	try{
	  dsToHnlMu_kinTree = dsToHnlMu_vertexFitter.fit(dsParticles); 
	}
	catch(VertexException eee){
	  continue;
	}

	if (!dsToHnlMu_kinTree->isValid()) continue;

	dsToHnlMu_kinTree->movePointerToTheTop();
	RefCountedKinematicParticle ds_particle = dsToHnlMu_kinTree->currentParticle();
	RefCountedKinematicVertex   ds_vtx      = dsToHnlMu_kinTree->currentDecayVertex();

	double fitted_ds_mass = ds_particle->currentState().mass();


	double ds_vtxprob = TMath::Prob(ds_vtx->chiSquared(), ds_vtx->degreesOfFreedom());
	if(ds_vtxprob < vtx_prob_cut_) continue;

	bool is_hnl_brother = false;
	bool isMCMatchedDsMuon = false;

	//mc matching for signal particles
	if(run==1 && isSignal){
	  int match_dsMu_idx = getMatchedGenPartIdx(iMuonDs->pt(),iMuonDs->eta(),iMuonDs->phi(),13,*packedGenParticleCollection);
	  if (match_dsMu_idx>0){
	    isMCMatchedDsMuon = true;
	    pat::PackedGenParticle matchedGenMuon = (*packedGenParticleCollection).at(match_dsMu_idx);
	    if (matchedGenMuon.motherRef().isNonnull() &&
		matchedGenMuon.motherRef().isAvailable()){
	      const reco::Candidate* genMuonMom = matchedGenMuon.mother(0);
	      for (unsigned i=0; i<genMuonMom->numberOfDaughters(); ++i){
		if(std::abs(genMuonMom->daughter(i)->pdgId()) == 9900015){
		  is_hnl_brother = true;
		  break;
		}
	      }
	    }
	  }
	}


	//get hnl pt
	Double_t hnl_px = hnl_particle->currentState().globalMomentum().x();
	Double_t hnl_py = hnl_particle->currentState().globalMomentum().y();
	Double_t hnl_pz = hnl_particle->currentState().globalMomentum().z();
	Double_t hnl_pt = TMath::Sqrt(hnl_px*hnl_px + hnl_py*hnl_py);

	//compute hnl cos2D and cos3D wrt beam spot
	Double_t hnl_dx = hnl_vtx->position().x() - (*theBeamSpot).position().x();
	Double_t hnl_dy = hnl_vtx->position().y() - (*theBeamSpot).position().y();
	Double_t hnl_dz = hnl_vtx->position().z() - (*theBeamSpot).position().z();
	Double_t cos3D_hnl = (hnl_px*hnl_dx + hnl_py*hnl_dy + hnl_pz*hnl_dz)/(sqrt(hnl_dx*hnl_dx + hnl_dy*hnl_dy + hnl_dz*hnl_dz)*hnl_particle->currentState().globalMomentum().mag());
	Double_t cos2D_hnl = (hnl_px*hnl_dx + hnl_py*hnl_dy)/(sqrt(hnl_dx*hnl_dx + hnl_dy*hnl_dy)*sqrt(hnl_px*hnl_px + hnl_py*hnl_py));

	// get hnl vertex info
	Double_t hnl_vx    = hnl_vtx->position().x();
	Double_t hnl_vy    = hnl_vtx->position().y();
	Double_t hnl_vz    = hnl_vtx->position().z();
	Double_t hnl_vxErr = hnl_vtx->error().cxx();
	Double_t hnl_vyErr = hnl_vtx->error().cyy();
	Double_t hnl_vzErr = hnl_vtx->error().czz();

	// get Ds vertex info
	Double_t ds_vx    = ds_vtx->position().x();
	Double_t ds_vy    = ds_vtx->position().y();
	Double_t ds_vz    = ds_vtx->position().z();
	Double_t ds_hnl_vxErr = ds_vtx->error().cxx();
	Double_t ds_hnl_vyErr = ds_vtx->error().cyy();
	Double_t ds_hnl_vzErr = ds_vtx->error().czz();

	//get Ds pt
	Double_t px_ds = ds_particle->currentState().globalMomentum().x();
	Double_t py_ds = ds_particle->currentState().globalMomentum().y();
	Double_t pz_ds = ds_particle->currentState().globalMomentum().z();
	Double_t p_ds = ds_particle->currentState().globalMomentum().mag();
	Double_t pt_ds = TMath::Sqrt(px_ds*px_ds + py_ds*py_ds);

	//compute ds cos2D and cos3D wrt beam spot
	Double_t dx_ds = ds_vtx->position().x() - (*theBeamSpot).position().x();
	Double_t dy_ds = ds_vtx->position().y() - (*theBeamSpot).position().y();
	Double_t dz_ds = ds_vtx->position().z() - (*theBeamSpot).position().z();
	Double_t cos3D_ds = (px_ds*dx_ds + py_ds*dy_ds + pz_ds*dz_ds)/(sqrt(dx_ds*dx_ds + dy_ds*dy_ds + dz_ds*dz_ds)*p_ds);
	Double_t cos2D_ds = (px_ds*dx_ds + py_ds*dy_ds)/(sqrt(dx_ds*dx_ds + dy_ds*dy_ds)*sqrt(px_ds*px_ds + py_ds*py_ds));

	//   SAVE
	C_Ds_vertex_prob ->push_back(ds_vtxprob);
	C_Ds_mass ->push_back(fitted_ds_mass);
	C_Ds_preFit_mass ->push_back((p4pi + p4muhnl + p4muds).M());
	C_Ds_px ->push_back(px_ds);
	C_Ds_py ->push_back(py_ds);
	C_Ds_pz ->push_back(pz_ds);
	C_Ds_pt ->push_back(pt_ds);
	C_Ds_vertex_x   ->push_back(ds_vx);
	C_Ds_vertex_y   ->push_back(ds_vy);
	C_Ds_vertex_z   ->push_back(ds_vz);
	C_Ds_vertex_xErr->push_back(ds_hnl_vxErr);
	C_Ds_vertex_yErr->push_back(ds_hnl_vyErr);
	C_Ds_vertex_zErr->push_back(ds_hnl_vzErr);
	C_Ds_vertex_sig->push_back(sqrt(ds_vx*ds_vx + ds_vy*ds_vy + ds_vz*ds_vz)/sqrt(ds_hnl_vxErr*ds_hnl_vxErr + ds_hnl_vyErr*ds_hnl_vyErr + ds_hnl_vzErr*ds_hnl_vzErr));
	C_Ds_vertex_cos3D->push_back(cos3D_ds);
	C_Ds_vertex_cos2D->push_back(cos2D_ds);

	C_Hnl_vertex_prob ->push_back(hnl_vtxprob);
	C_Hnl_mass ->push_back(hnl_mass);
	C_Hnl_preFit_mass ->push_back((p4muhnl + p4pi).M());
	C_Hnl_px ->push_back(hnl_px);
	C_Hnl_py ->push_back(hnl_py);
	C_Hnl_pz ->push_back(hnl_pz);
	C_Hnl_pt ->push_back(hnl_pt);
	C_Hnl_vertex_x   ->push_back(hnl_vx);
	C_Hnl_vertex_y   ->push_back(hnl_vy);
	C_Hnl_vertex_z   ->push_back(hnl_vz);
	C_Hnl_vertex_xErr->push_back(hnl_vxErr);
	C_Hnl_vertex_yErr->push_back(hnl_vyErr);
	C_Hnl_vertex_zErr->push_back(hnl_vzErr);
	C_Hnl_vertex_sig->push_back(sqrt(hnl_vx*hnl_vx + hnl_vy*hnl_vy + hnl_vz*hnl_vz)/sqrt(hnl_vxErr*hnl_vxErr + hnl_vyErr*hnl_vyErr + hnl_vzErr*hnl_vzErr));
	C_Hnl_vertex_cos3D->push_back(cos3D_hnl);
	C_Hnl_vertex_cos2D->push_back(cos2D_hnl);

	C_mu_Hnl_px ->push_back(p4muhnl.Px());
	C_mu_Hnl_py ->push_back(p4muhnl.Py());
	C_mu_Hnl_pz ->push_back(p4muhnl.Pz());
	C_mu_Hnl_pt ->push_back(p4muhnl.Pt());
	C_mu_Hnl_eta ->push_back(p4muhnl.Eta());
	C_mu_Hnl_phi ->push_back(p4muhnl.Phi());
	C_mu_Hnl_BS_ips_xy ->push_back(iMuonHnl->dB(pat::Muon::BS2D)/iMuonHnl->edB(pat::Muon::BS2D));
	C_mu_Hnl_BS_ips    ->push_back(iMuonHnl->dB(pat::Muon::BS3D)/iMuonHnl->edB(pat::Muon::BS3D));
	C_mu_Hnl_PV_ips_z  ->push_back(iMuonHnl->dB(pat::Muon::PVDZ)/iMuonHnl->edB(pat::Muon::PVDZ));
	C_mu_Hnl_BS_ip_xy  ->push_back(iMuonHnl->dB(pat::Muon::BS2D));
	C_mu_Hnl_BS_ip     ->push_back(iMuonHnl->dB(pat::Muon::BS3D));
	C_mu_Hnl_PV_ip_z   ->push_back(iMuonHnl->dB(pat::Muon::PVDZ));
	C_mu_Hnl_isHnlDaughter->push_back(hnl_mu_match ? 1 : 0);
	C_mu_Hnl_charge->push_back(iMuonHnl->charge());
	C_mu_Hnl_isSoft   ->push_back(isSoftMuonHnl? 1: 0);
	C_mu_Hnl_isLoose  ->push_back(isLooseMuonHnl? 1: 0);
	C_mu_Hnl_isMedium ->push_back(isMediumMuonHnl? 1: 0);
	C_mu_Hnl_isGlobal   ->push_back(isGlobalMuonHnl? 1: 0);
	C_mu_Hnl_isTracker  ->push_back(isTrackerMuonHnl? 1: 0);
	C_mu_Hnl_isStandAlone ->push_back(isStandAloneMuonHnl? 1: 0);
	C_mu_Hnl_isMCMatched ->push_back(isMCMatchedMuonHnl? 1: 0);
	C_mu_Hnl_idx ->push_back(i_muhnl);


	C_mu_Ds_px ->push_back(p4muds.Px());
	C_mu_Ds_py ->push_back(p4muds.Py());
	C_mu_Ds_pz ->push_back(p4muds.Pz());
	C_mu_Ds_pt ->push_back(p4muds.Pt());
	C_mu_Ds_eta ->push_back(p4muds.Eta());
	C_mu_Ds_phi ->push_back(p4muds.Phi());
	C_mu_Ds_BS_ips_xy ->push_back(iMuonDs->dB(pat::Muon::BS2D)/iMuonDs->edB(pat::Muon::BS2D));
	C_mu_Ds_BS_ips    ->push_back(iMuonDs->dB(pat::Muon::BS3D)/iMuonDs->edB(pat::Muon::BS3D));
	C_mu_Ds_PV_ips_z  ->push_back(iMuonDs->dB(pat::Muon::PVDZ)/iMuonDs->edB(pat::Muon::PVDZ));
	C_mu_Ds_BS_ip_xy  ->push_back(iMuonDs->dB(pat::Muon::BS2D));
	C_mu_Ds_BS_ip     ->push_back(iMuonDs->dB(pat::Muon::BS3D));
	C_mu_Ds_PV_ip_z   ->push_back(iMuonDs->dB(pat::Muon::PVDZ));
	C_mu_Ds_charge ->push_back(iMuonDs->charge());
	C_mu_Ds_isHnlBrother->push_back(is_hnl_brother ? 1 : 0);
	C_mu_Ds_isSoft   ->push_back(isSoftMuonDs? 1: 0);
	C_mu_Ds_isLoose  ->push_back(isLooseMuonDs? 1: 0);
	C_mu_Ds_isMedium ->push_back(isMediumMuonDs? 1: 0);
	C_mu_Ds_isGlobal   ->push_back(isGlobalMuonDs? 1: 0);
	C_mu_Ds_isTracker  ->push_back(isTrackerMuonDs? 1: 0);
	C_mu_Ds_isStandAlone ->push_back(isStandAloneMuonDs? 1: 0);
	C_mu_Ds_isMCMatched ->push_back(isMCMatchedDsMuon? 1: 0);
	C_mu_Ds_idx ->push_back(i_muds);

	C_pi_charge ->push_back(iTrack1->charge());
	C_pi_px ->push_back(p4pi.Px());
	C_pi_py ->push_back(p4pi.Py());
	C_pi_pz ->push_back(p4pi.Pz());
	C_pi_pt ->push_back(p4pi.Pt());
	C_pi_eta ->push_back(p4pi.Eta());
	C_pi_phi ->push_back(p4pi.Phi());
	C_pi_BS_ips_xy->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position()))/std::abs(iTrack1->dxyError()));
	C_pi_BS_ips_z ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position()))/std::abs(iTrack1->dzError()));
	C_pi_BS_ip_xy ->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position())));
	C_pi_BS_ip_z  ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position())));
	C_pi_isMCMatched ->push_back(isMCMatchedTrack1? 1: 0);
	C_pi_isHnlDaughter->push_back(hnl_C_pi_match ? 1 : 0);

	C_mu1mu2_mass->push_back((p4muhnl + p4muds).M());
	C_mu1mu2_dr->push_back(deltaR(*iMuonHnl,*iMuonDs));
	C_mu1pi_dr->push_back(deltaR(*iMuonHnl,*iTrack1));
	C_mu2pi_dr->push_back(deltaR(*iMuonDs,*iTrack1));

	++nCand;

	//hnlParticles.clear();
      } 
    } 
  } 


  // ===================== END OF EVENT : WRITE ETC ++++++++++++++++++++++

  if (nCand > 0)
  {
    cout << "_____________________ SUCCESS!!!! _______________________" << endl;
    ++N_written_events;
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

  C_Hnl_vertex_prob->clear();
  C_Hnl_mass->clear(); 
  C_Hnl_preFit_mass->clear(); 
  C_Hnl_px->clear();
  C_Hnl_py->clear();
  C_Hnl_pz->clear();
  C_Hnl_pt->clear();
  C_Hnl_vertex_x->clear();
  C_Hnl_vertex_y->clear();
  C_Hnl_vertex_z->clear();
  C_Hnl_vertex_xErr->clear();
  C_Hnl_vertex_yErr->clear();
  C_Hnl_vertex_zErr->clear();
  C_Hnl_vertex_sig->clear();
  C_Hnl_vertex_cos3D->clear();
  C_Hnl_vertex_cos2D->clear();
  //
  C_mu_Hnl_px->clear();
  C_mu_Hnl_py->clear();
  C_mu_Hnl_pz->clear();
  C_mu_Hnl_pt->clear();
  C_mu_Hnl_eta->clear();
  C_mu_Hnl_phi->clear();
  C_mu_Hnl_BS_ips_xy->clear();
  C_mu_Hnl_BS_ips->clear();
  C_mu_Hnl_PV_ips_z->clear();
  C_mu_Hnl_BS_ip_xy->clear();
  C_mu_Hnl_BS_ip->clear();
  C_mu_Hnl_PV_ip_z->clear();
  C_mu_Hnl_charge->clear();
  C_mu_Hnl_isSoft->clear();
  C_mu_Hnl_isLoose->clear();
  C_mu_Hnl_isMedium->clear();
  C_mu_Hnl_isGlobal->clear();
  C_mu_Hnl_isTracker->clear();
  C_mu_Hnl_isStandAlone->clear();
  C_mu_Hnl_isMCMatched->clear();
  C_mu_Hnl_idx->clear();

  C_mu_Ds_px->clear();
  C_mu_Ds_py->clear();
  C_mu_Ds_pz->clear();
  C_mu_Ds_pt->clear();
  C_mu_Ds_eta->clear();
  C_mu_Ds_phi->clear();
  C_mu_Ds_BS_ips_xy->clear();
  C_mu_Ds_BS_ips->clear();
  C_mu_Ds_PV_ips_z->clear();
  C_mu_Ds_BS_ip_xy->clear();
  C_mu_Ds_BS_ip->clear();
  C_mu_Ds_PV_ip_z->clear();
  C_mu_Ds_charge->clear();
  C_mu_Ds_isSoft->clear();
  C_mu_Ds_isLoose->clear();
  C_mu_Ds_isMedium->clear();
  C_mu_Ds_isGlobal->clear();
  C_mu_Ds_isTracker->clear();
  C_mu_Ds_isStandAlone->clear();
  C_mu_Ds_isMCMatched->clear();
  C_mu_Ds_idx->clear();

  C_mu1mu2_mass->clear();
  C_mu1mu2_dr->clear();
  C_mu1pi_dr->clear();
  C_mu2pi_dr->clear();

  C_pi_charge->clear();
  C_pi_px->clear(); 
  C_pi_py->clear();
  C_pi_pz->clear();
  C_pi_pt->clear();
  C_pi_eta->clear();
  C_pi_phi->clear();
  C_pi_BS_ips_xy->clear();
  C_pi_BS_ips_z->clear();
  C_pi_BS_ip_xy->clear();
  C_pi_BS_ip_z->clear();
  C_pi_isMCMatched->clear();

  PV_x->clear();   PV_y->clear();   PV_z->clear();
  PV_xErr->clear();   PV_yErr->clear();   PV_zErr->clear();
  PV_prob->clear();   //PV_dN->clear();

  HLT_mu_trig_idx->clear();
  HLT_mu_trig_pt->clear();
  HLT_mu_trig_eta->clear();

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

  C_pi_isHnlDaughter->clear();
  C_mu_Hnl_isHnlDaughter->clear();
  C_mu_Ds_isHnlBrother->clear();


}

template <typename A,typename B> bool DsToHnlMu_HnlToMuPi_prompt_miniAOD::IsTheSame(const A& cand1, const B& cand2){
  double deltaPt  = std::abs(cand1.pt()-cand2.pt());
  double deltaEta = cand1.eta()-cand2.eta();

  auto deltaPhi = std::abs(cand1.phi() - cand2.phi());
  if (deltaPhi > float(M_PI))
    deltaPhi -= float(2 * M_PI);
  double deltaR2 = deltaEta*deltaEta + deltaPhi*deltaPhi;
  if(deltaR2<0.01 && deltaPt<0.1) return true;
  else return false;
}

int DsToHnlMu_HnlToMuPi_prompt_miniAOD::getMatchedGenPartIdx(double pt, double eta, double phi, int pdg_id, std::vector<pat::PackedGenParticle> packedGen_particles){
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
    if (deltaPt<0.5 && deltaR2<0.09 && deltaR2<max_dr2) {
      matchedIndex = i;
      max_dr2 = deltaR2;
    }
  }

  return matchedIndex;  
}


// ------------ method called once each job just before starting event loop  ------------
void DsToHnlMu_HnlToMuPi_prompt_miniAOD::beginJob()
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


  wwtree->Branch("C_Hnl_vertex_prob", &C_Hnl_vertex_prob);
  wwtree->Branch("C_Hnl_mass", &C_Hnl_mass);
  wwtree->Branch("C_Hnl_preFit_mass", &C_Hnl_preFit_mass);
  wwtree->Branch("C_Hnl_px", &C_Hnl_px);
  wwtree->Branch("C_Hnl_py", &C_Hnl_py);
  wwtree->Branch("C_Hnl_pz", &C_Hnl_pz);
  wwtree->Branch("C_Hnl_pt", &C_Hnl_pt);
  wwtree->Branch("C_Hnl_vertex_x" , &C_Hnl_vertex_x);
  wwtree->Branch("C_Hnl_vertex_y" , &C_Hnl_vertex_y);
  wwtree->Branch("C_Hnl_vertex_z" , &C_Hnl_vertex_z);
  wwtree->Branch("C_Hnl_vertex_xErr" , &C_Hnl_vertex_xErr);
  wwtree->Branch("C_Hnl_vertex_yErr" , &C_Hnl_vertex_yErr);
  wwtree->Branch("C_Hnl_vertex_zErr" , &C_Hnl_vertex_zErr);
  wwtree->Branch("C_Hnl_vertex_sig" , &C_Hnl_vertex_sig);
  wwtree->Branch("C_Hnl_vertex_cos3D" , &C_Hnl_vertex_cos3D);
  wwtree->Branch("C_Hnl_vertex_cos2D" , &C_Hnl_vertex_cos2D);

  wwtree->Branch("C_mu_Hnl_px"    , &C_mu_Hnl_px);
  wwtree->Branch("C_mu_Hnl_py"    , &C_mu_Hnl_py);
  wwtree->Branch("C_mu_Hnl_pz"    , &C_mu_Hnl_pz);
  wwtree->Branch("C_mu_Hnl_pt"    , &C_mu_Hnl_pt);
  wwtree->Branch("C_mu_Hnl_eta"   , &C_mu_Hnl_eta);
  wwtree->Branch("C_mu_Hnl_phi"   , &C_mu_Hnl_phi);
  wwtree->Branch("C_mu_Hnl_BS_ips_xy", &C_mu_Hnl_BS_ips_xy);
  wwtree->Branch("C_mu_Hnl_BS_ips", &C_mu_Hnl_BS_ips);
  wwtree->Branch("C_mu_Hnl_PV_ips_z" , &C_mu_Hnl_PV_ips_z);
  wwtree->Branch("C_mu_Hnl_BS_ip_xy" , &C_mu_Hnl_BS_ip_xy);
  wwtree->Branch("C_mu_Hnl_BS_ip" , &C_mu_Hnl_BS_ip);
  wwtree->Branch("C_mu_Hnl_PV_ip_z"  , &C_mu_Hnl_PV_ip_z);
  wwtree->Branch("C_mu_Hnl_charge", &C_mu_Hnl_charge);
  wwtree->Branch("C_mu_Hnl_isSoft", &C_mu_Hnl_isSoft);
  wwtree->Branch("C_mu_Hnl_isLoose", &C_mu_Hnl_isLoose);
  wwtree->Branch("C_mu_Hnl_isMedium", &C_mu_Hnl_isMedium);
  wwtree->Branch("C_mu_Hnl_isGlobal", &C_mu_Hnl_isGlobal);
  wwtree->Branch("C_mu_Hnl_isTracker", &C_mu_Hnl_isTracker);
  wwtree->Branch("C_mu_Hnl_isStandAlone", &C_mu_Hnl_isStandAlone);
  wwtree->Branch("C_mu_Hnl_isMCMatched", &C_mu_Hnl_isMCMatched);
  wwtree->Branch("C_mu_Hnl_idx", &C_mu_Hnl_idx);

  wwtree->Branch("C_mu_Ds_px"    , &C_mu_Ds_px);
  wwtree->Branch("C_mu_Ds_py"    , &C_mu_Ds_py);
  wwtree->Branch("C_mu_Ds_pz"    , &C_mu_Ds_pz);
  wwtree->Branch("C_mu_Ds_pt"    , &C_mu_Ds_pt);
  wwtree->Branch("C_mu_Ds_eta"   , &C_mu_Ds_eta);
  wwtree->Branch("C_mu_Ds_phi"   , &C_mu_Ds_phi);
  wwtree->Branch("C_mu_Ds_BS_ips_xy", &C_mu_Ds_BS_ips_xy);
  wwtree->Branch("C_mu_Ds_BS_ips", &C_mu_Ds_BS_ips);
  wwtree->Branch("C_mu_Ds_PV_ips_z" , &C_mu_Ds_PV_ips_z);
  wwtree->Branch("C_mu_Ds_BS_ip_xy" , &C_mu_Ds_BS_ip_xy);
  wwtree->Branch("C_mu_Ds_BS_ip" , &C_mu_Ds_BS_ip);
  wwtree->Branch("C_mu_Ds_PV_ip_z"  , &C_mu_Ds_PV_ip_z);
  wwtree->Branch("C_mu_Ds_charge", &C_mu_Ds_charge);
  wwtree->Branch("C_mu_Ds_isSoft", &C_mu_Ds_isSoft);
  wwtree->Branch("C_mu_Ds_isLoose", &C_mu_Ds_isLoose);
  wwtree->Branch("C_mu_Ds_isMedium", &C_mu_Ds_isMedium);
  wwtree->Branch("C_mu_Ds_isGlobal", &C_mu_Ds_isGlobal);
  wwtree->Branch("C_mu_Ds_isTracker", &C_mu_Ds_isTracker);
  wwtree->Branch("C_mu_Ds_isStandAlone", &C_mu_Ds_isStandAlone);
  wwtree->Branch("C_mu_Ds_isMCMatched", &C_mu_Ds_isMCMatched);
  wwtree->Branch("C_mu_Ds_idx", &C_mu_Ds_idx);

  wwtree->Branch("C_mu1mu2_mass"     , &C_mu1mu2_mass   );
  wwtree->Branch("C_mu1mu2_dr"       , &C_mu1mu2_dr   );
  wwtree->Branch("C_mu1pi_dr"        , &C_mu1pi_dr   );
  wwtree->Branch("C_mu2pi_dr"        , &C_mu2pi_dr   );

  wwtree->Branch("C_pi_charge"       , &C_pi_charge     );
  wwtree->Branch("C_pi_px"           , &C_pi_px         );
  wwtree->Branch("C_pi_py"           , &C_pi_py         );
  wwtree->Branch("C_pi_pz"           , &C_pi_pz         );
  wwtree->Branch("C_pi_pt"           , &C_pi_pt         );
  wwtree->Branch("C_pi_eta"          , &C_pi_eta        );
  wwtree->Branch("C_pi_phi"          , &C_pi_phi        );
  wwtree->Branch("C_pi_BS_ips_xy"       , &C_pi_BS_ips_xy     );
  wwtree->Branch("C_pi_BS_ips_z"        , &C_pi_BS_ips_z     );
  wwtree->Branch("C_pi_BS_ip_xy"        , &C_pi_BS_ip_xy     );
  wwtree->Branch("C_pi_BS_ip_z"         , &C_pi_BS_ip_z     );
  wwtree->Branch("C_pi_isMCMatched", &C_pi_isMCMatched);

  wwtree->Branch("PV_x"    , &PV_x   );
  wwtree->Branch("PV_y"    , &PV_y   );
  wwtree->Branch("PV_z"    , &PV_z   );
  wwtree->Branch("PV_xErr" , &PV_xErr);
  wwtree->Branch("PV_yErr" , &PV_yErr);
  wwtree->Branch("PV_zErr" , &PV_zErr);
  wwtree->Branch("PV_prob" , &PV_prob);
  //wwtree->Branch("PV_dN"   , &PV_dN);

  wwtree->Branch("HLT_mu_trig_idx",    &HLT_mu_trig_idx);
  wwtree->Branch("HLT_mu_trig_pt",    &HLT_mu_trig_pt);
  wwtree->Branch("HLT_mu_trig_eta",    &HLT_mu_trig_eta);

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

  wwtree->Branch("C_pi_isHnlDaughter"      , &C_pi_isHnlDaughter );
  wwtree->Branch("C_mu_Hnl_isHnlDaughter"  , &C_mu_Hnl_isHnlDaughter );
  wwtree->Branch("C_mu_Ds_isHnlBrother"    , &C_mu_Ds_isHnlBrother );

}

// ------------ method called once each job just after ending the event loop  ------------
  void
DsToHnlMu_HnlToMuPi_prompt_miniAOD::endJob()
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
DsToHnlMu_HnlToMuPi_prompt_miniAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DsToHnlMu_HnlToMuPi_prompt_miniAOD);
