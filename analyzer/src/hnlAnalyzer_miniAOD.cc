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
class hnlAnalyzer_miniAOD : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
  public:
    explicit hnlAnalyzer_miniAOD(const edm::ParameterSet&);
    ~hnlAnalyzer_miniAOD();

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

    Int_t N_written_events;

    std::vector<float> *C_Hnl_vertex_prob;
    std::vector<float> *C_Hnl_mass;
    std::vector<float> *C_Hnl_preFit_mass;
    std::vector<float> *C_Hnl_px, *C_Hnl_py, *C_Hnl_pz;
    std::vector<float> *C_Hnl_vertex_x, *C_Hnl_vertex_y, *C_Hnl_vertex_z;
    std::vector<float> *C_Hnl_vertex_xErr, *C_Hnl_vertex_yErr, *C_Hnl_vertex_zErr;
    std::vector<float> *C_Hnl_vertex_sig;
    std::vector<float> *C_Hnl_vertex_cos3D;
    std::vector<float> *C_Hnl_vertex_cos2D;

    std::vector<float> *C_mu1_px, *C_mu1_py, *C_mu1_pz;
    std::vector<float> *C_mu1_eta;
    std::vector<float> *C_mu1_phi;
    std::vector<float> *C_mu1_BS_ips_xy, *C_mu1_PV_ips_z;
    std::vector<float> *C_mu1_BS_ips;
    std::vector<float> *C_mu1_BS_ip_xy, *C_mu1_PV_ip_z;
    std::vector<float> *C_mu1_BS_ip;
    std::vector<int>   *C_mu1_charge;
    std::vector<short> *C_mu1_isSoft;
    std::vector<short> *C_mu1_isLoose;
    std::vector<short> *C_mu1_isMedium;
    std::vector<short> *C_mu1_isGlobal;
    std::vector<short> *C_mu1_isTracker;
    std::vector<short> *C_mu1_isStandAlone;
    std::vector<short> *C_mu1_isMCMatched;

    std::vector<float> *C_trigMu_px, *C_trigMu_py, *C_trigMu_pz;
    std::vector<float> *C_trigMu_eta;
    std::vector<float> *C_trigMu_phi;
    std::vector<float> *C_trigMu_BS_ips_xy, *C_trigMu_PV_ips_z;
    std::vector<float> *C_trigMu_BS_ips;
    std::vector<float> *C_trigMu_BS_ip_xy, *C_trigMu_PV_ip_z;
    std::vector<float> *C_trigMu_BS_ip;
    std::vector<int>   *C_trigMu_charge;
    std::vector<short> *C_trigMu_isSoft;
    std::vector<short> *C_trigMu_isLoose;
    std::vector<short> *C_trigMu_isMedium;
    std::vector<short> *C_trigMu_isGlobal;
    std::vector<short> *C_trigMu_isTracker;
    std::vector<short> *C_trigMu_isStandAlone;
    std::vector<short> *C_trigMu_isMCMatched;

    std::vector<float> *C_mass;
    std::vector<float> *C_mu1mu2_mass;
    std::vector<float> *C_mu1mu2_dr;
    std::vector<float> *C_mu1pi_dr;
    std::vector<float> *C_mu2pi_dr;
    std::vector<int>   *C_pi_charge;
    std::vector<float> *C_px, *C_py, *C_pz;
    std::vector<float> *C_pi_px, *C_pi_py, *C_pi_pz;
    std::vector<float> *C_pi_eta;
    std::vector<float> *C_pi_phi;
    std::vector<float> *C_pi_BS_ips_xy, *C_pi_BS_ips_z;
    std::vector<float> *C_pi_BS_ip_xy, *C_pi_BS_ip_z;
    std::vector<short> *C_pi_isMCMatched;

    std::vector<float> *PV_x, *PV_y, *PV_z;
    std::vector<float> *PV_xErr, *PV_yErr, *PV_zErr;
    std::vector<float> *PV_prob;
    //std::vector<int>   *PV_dN;

    std::vector<short>  *mu7_ip4_matched;
    std::vector<short>  *mu8_ip3_matched;
    std::vector<short>  *mu8_ip3p5_matched ;
    std::vector<short>  *mu8_ip5_matched;
    std::vector<short>  *mu8_ip6_matched;
    std::vector<short>  *mu9_ip4_matched;
    std::vector<short>  *mu9_ip5_matched;
    std::vector<short>  *mu9_ip6_matched;
    std::vector<short>  *mu10p5_ip3p5_matched;
    std::vector<short>  *mu12_ip6_matched;

    std::vector<float>  *mu7_ip4_eta;
    std::vector<float>  *mu8_ip3_eta;
    std::vector<float>  *mu8_ip3p5_eta;
    std::vector<float>  *mu8_ip5_eta;
    std::vector<float>  *mu8_ip6_eta;
    std::vector<float>  *mu9_ip4_eta;
    std::vector<float>  *mu9_ip5_eta;
    std::vector<float>  *mu9_ip6_eta;
    std::vector<float>  *mu10p5_ip3p5_eta;
    std::vector<float>  *mu12_ip6_eta;

    std::vector<float>  *mu7_ip4_pt;
    std::vector<float>  *mu8_ip3_pt;
    std::vector<float>  *mu8_ip3p5_pt ;
    std::vector<float>  *mu8_ip5_pt;
    std::vector<float>  *mu8_ip6_pt;
    std::vector<float>  *mu9_ip4_pt;
    std::vector<float>  *mu9_ip5_pt;
    std::vector<float>  *mu9_ip6_pt;
    std::vector<float>  *mu10p5_ip3p5_pt;
    std::vector<float>  *mu12_ip6_pt;

    std::vector<float>  *mu7_ip4_dr;
    std::vector<float>  *mu8_ip3_dr;
    std::vector<float>  *mu8_ip3p5_dr ;
    std::vector<float>  *mu8_ip5_dr;
    std::vector<float>  *mu8_ip6_dr;
    std::vector<float>  *mu9_ip4_dr;
    std::vector<float>  *mu9_ip5_dr;
    std::vector<float>  *mu9_ip6_dr;
    std::vector<float>  *mu10p5_ip3p5_dr;
    std::vector<float>  *mu12_ip6_dr;

    std::vector<short>  *mu7_ip4_fired;
    std::vector<short>  *mu8_ip3_fired;
    std::vector<short>  *mu8_ip3p5_fired;
    std::vector<short>  *mu8_ip5_fired;
    std::vector<short>  *mu8_ip6_fired;
    std::vector<short>  *mu9_ip4_fired;
    std::vector<short>  *mu9_ip5_fired;
    std::vector<short>  *mu9_ip6_fired;
    std::vector<short>  *mu10p5_ip3p5_fired;
    std::vector<short>  *mu12_ip6_fired;


    std::vector<short>  *C_pi_isHnlDaughter;
    std::vector<short>  *C_mu1_isHnlDaughter;
    std::vector<short>  *C_trigMu_isHnlBrother;

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


hnlAnalyzer_miniAOD::hnlAnalyzer_miniAOD(const edm::ParameterSet& iConfig) :
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
  trigMu_pt_cut_     (iConfig.getUntrackedParameter<double>("trigMu_pt_cut" )),
  trigMu_eta_cut_    (iConfig.getUntrackedParameter<double>("trigMu_eta_cut")),
  pi_pt_cut_         (iConfig.getUntrackedParameter<double>("pi_pt_cut" )),
  pi_eta_cut_        (iConfig.getUntrackedParameter<double>("pi_eta_cut")),
  mumupi_mass_cut_   (iConfig.getUntrackedParameter<double>("b_mass_cut")),
  mupi_mass_high_cut_(iConfig.getUntrackedParameter<double>("mupi_mass_high_cut")),
  mupi_mass_low_cut_ (iConfig.getUntrackedParameter<double>("mupi_mass_low_cut")),
  mupi_pt_cut_       (iConfig.getUntrackedParameter<double>("mupi_pt_cut")),
  vtx_prob_cut_      (iConfig.getUntrackedParameter<double>("vtx_prob_cut")),

  C_Hnl_vertex_prob(0),
  C_Hnl_mass(0),
  C_Hnl_preFit_mass(0),
  C_Hnl_px(0),
  C_Hnl_py(0),  
  C_Hnl_pz(0),
  C_Hnl_vertex_x(0),  
  C_Hnl_vertex_y(0), 
  C_Hnl_vertex_z(0),
  C_Hnl_vertex_xErr(0),  
  C_Hnl_vertex_yErr(0), 
  C_Hnl_vertex_zErr(0),
  C_Hnl_vertex_sig(0),
  C_Hnl_vertex_cos3D(0),
  C_Hnl_vertex_cos2D(0),

  C_mu1_px(0),
  C_mu1_py(0),       
  C_mu1_pz(0),
  C_mu1_eta(0),
  C_mu1_phi(0),
  C_mu1_BS_ips_xy(0),
  C_mu1_PV_ips_z(0),
  C_mu1_BS_ips(0),
  C_mu1_BS_ip_xy(0),
  C_mu1_PV_ip_z(0),
  C_mu1_BS_ip(0),
  C_mu1_charge(0),
  C_mu1_isSoft(0),
  C_mu1_isLoose(0),
  C_mu1_isMedium(0),
  C_mu1_isGlobal(0),
  C_mu1_isTracker(0),
  C_mu1_isStandAlone(0),
  C_mu1_isMCMatched(0),

  C_trigMu_px(0),
  C_trigMu_py(0),
  C_trigMu_pz(0),
  C_trigMu_eta(0),
  C_trigMu_phi(0),
  C_trigMu_BS_ips_xy(0),
  C_trigMu_PV_ips_z(0),
  C_trigMu_BS_ips(0),
  C_trigMu_BS_ip_xy(0),
  C_trigMu_PV_ip_z(0),
  C_trigMu_BS_ip(0),
  C_trigMu_charge(0),
  C_trigMu_isSoft(0),
  C_trigMu_isLoose(0),
  C_trigMu_isMedium(0),
  C_trigMu_isGlobal(0),
  C_trigMu_isTracker(0),
  C_trigMu_isStandAlone(0),
  C_trigMu_isMCMatched(0),

  C_mass(0),
  C_mu1mu2_mass(0),
  C_mu1mu2_dr(0),
  C_mu1pi_dr(0),
  C_mu2pi_dr(0),
  C_pi_charge(0),
  C_px(0),
  C_py(0),
  C_pz(0),
  C_pi_px(0),
  C_pi_py(0),
  C_pi_pz(0),
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

  mu7_ip4_matched(0),
  mu8_ip3_matched(0),
  mu8_ip3p5_matched (0),
  mu8_ip5_matched(0),
  mu8_ip6_matched(0),
  mu9_ip4_matched(0),
  mu9_ip5_matched(0),
  mu9_ip6_matched(0),
  mu10p5_ip3p5_matched(0),
  mu12_ip6_matched(0),

  mu7_ip4_eta(0),
  mu8_ip3_eta(0),
  mu8_ip3p5_eta (0),
  mu8_ip5_eta(0),
  mu8_ip6_eta(0),
  mu9_ip4_eta(0),
  mu9_ip5_eta(0),
  mu9_ip6_eta(0),
  mu10p5_ip3p5_eta(0),
  mu12_ip6_eta(0),

  mu7_ip4_pt(0),
  mu8_ip3_pt(0),
  mu8_ip3p5_pt (0),
  mu8_ip5_pt(0),
  mu8_ip6_pt(0),
  mu9_ip4_pt(0),
  mu9_ip5_pt(0),
  mu9_ip6_pt(0),
  mu10p5_ip3p5_pt(0),
  mu12_ip6_pt(0),

  mu7_ip4_dr(0),
  mu8_ip3_dr(0),
  mu8_ip3p5_dr (0),
  mu8_ip5_dr(0),
  mu8_ip6_dr(0),
  mu9_ip4_dr(0),
  mu9_ip5_dr(0),
  mu9_ip6_dr(0),
  mu10p5_ip3p5_dr(0),
  mu12_ip6_dr(0),

  mu7_ip4_fired(0),
  mu8_ip3_fired(0),
  mu8_ip3p5_fired (0),
  mu8_ip5_fired(0),
  mu8_ip6_fired(0),
  mu9_ip4_fired(0),
  mu9_ip5_fired(0),
  mu9_ip6_fired(0),
  mu10p5_ip3p5_fired(0),
  mu12_ip6_fired(0),

  C_pi_isHnlDaughter(0),
  C_mu1_isHnlDaughter(0),
  C_trigMu_isHnlBrother(0),

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


hnlAnalyzer_miniAOD::~hnlAnalyzer_miniAOD()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
  void
hnlAnalyzer_miniAOD::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  std::vector<std::string> TriggerPaths = {
    "HLT_Mu7_IP4_part*" ,     // 0
    "HLT_Mu8_IP3_part*" ,     // 1
    "HLT_Mu8_IP3p5_part*" ,   // 2  
    "HLT_Mu8_IP5_part*" ,     // 3
    "HLT_Mu8_IP6_part*" ,     // 4  
    "HLT_Mu9_IP4_part*" ,     // 5
    "HLT_Mu9_IP5_part*" ,     // 6  
    "HLT_Mu9_IP6_part*" ,     // 7  
    "HLT_Mu10p5_IP3p5_part*", // 8
    "HLT_Mu12_IP6_part*"      // 9
  };

  unsigned int nTrigPaths = (unsigned int)TriggerPaths.size();

  std::vector<short> TriggersFired(nTrigPaths);
  std::vector<short> TriggerMatches(nTrigPaths);
  std::vector<float> TriggerPathPt(nTrigPaths);
  std::vector<float> TriggerPathEta(nTrigPaths);
  std::vector<float> TriggerPathDR(nTrigPaths);

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
  }
  else
  {
    std::cout << " No trigger Results in event :( " << run << "," << event << std::endl;
  }

  nCand= 0;

  KinematicParticleFactoryFromTransientTrack pFactory;


  //trigger muon
  //for ( std::vector<pat::Muon>::const_iterator iTrigMuon = thePATMuonHandle->begin(); iTrigMuon != thePATMuonHandle->end(); ++iTrigMuon){
  for ( unsigned i=0; i<thePATMuonHandle->size(); ++i){

    const pat::Muon* iTrigMuon = &(*thePATMuonHandle).at(i);

    //cut on muon pt and eta
    if(iTrigMuon->pt() < trigMu_pt_cut_) continue;
    if(std::abs(iTrigMuon->eta()) > trigMu_eta_cut_)  continue;

    //save muon id info
    bool isSoftTrigMuon = false;
    bool isLooseTrigMuon = false;
    bool isMediumTrigMuon = false;

    if (iTrigMuon->isSoftMuon(thePrimaryV)) isSoftTrigMuon=true;
    if (iTrigMuon->isLooseMuon())           isLooseTrigMuon=true;
    if (iTrigMuon->isMediumMuon())          isMediumTrigMuon=true;

    //save muon track info
    bool isGlobalTrigMuon = false;
    bool isTrackerTrigMuon = false;
    bool isStandAloneTrigMuon = false;

    if (iTrigMuon->isGlobalMuon())     isGlobalTrigMuon = true;
    if (iTrigMuon->isTrackerMuon())    isTrackerTrigMuon = true;
    if (iTrigMuon->isStandAloneMuon()) isStandAloneTrigMuon = true;


    TrackRef inTrackMu2;
    inTrackMu2 = iTrigMuon->track();
    if (inTrackMu2.isNull())  continue;
    if (!(inTrackMu2->quality(reco::TrackBase::highPurity)))  continue;

    TLorentzVector p4mu2;
    p4mu2.SetPtEtaPhiM(iTrigMuon->pt(), iTrigMuon->eta(), iTrigMuon->phi(), pdg.PDG_MUON_MASS);

    bool is_hnl_brother = false;
    bool isMCMatchedTrigMuon = false;

    //mc matching for signal particles
    if(run==1){
      int match_trigMu_idx = getMatchedGenPartIdx(iTrigMuon->pt(),iTrigMuon->eta(),iTrigMuon->phi(),13,*packedGenParticleCollection);

      if (match_trigMu_idx>0){
	isMCMatchedTrigMuon = true;
	pat::PackedGenParticle matchedGenMuon = (*packedGenParticleCollection).at(match_trigMu_idx);
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



    //for ( std::vector<pat::Muon>::const_iterator iMuon1 = thePATMuonHandle->begin(); iMuon1 != thePATMuonHandle->end(); ++iMuon1){
    for ( unsigned j=0; j<thePATMuonHandle->size(); ++j){
      if (i==j) continue;
    //for ( unsigned j=i+1; j<thePATMuonHandle->size(); ++j){


      //if (IsTheSame(*iMuon1,*iTrigMuon) ) continue;
      const pat::Muon* iMuon1 = &(*thePATMuonHandle).at(j);

      //cuts on muon pt and eta
      if (iMuon1->pt() < mu_pt_cut_) continue;
      if (std::abs(iMuon1->eta()) > mu_eta_cut_)  continue;

      //save muon id info
      bool isSoftMuon1 = false;
      bool isLooseMuon1 = false;
      bool isMediumMuon1 = false;

      if (iMuon1->isSoftMuon(thePrimaryV)) isSoftMuon1=true;
      if (iMuon1->isLooseMuon())           isLooseMuon1=true;
      if (iMuon1->isMediumMuon())          isMediumMuon1=true;

      //save muon track info
      bool isGlobalMuon1 = false;
      bool isTrackerMuon1 = false;
      bool isStandAloneMuon1 = false;

      if (iMuon1->isGlobalMuon())     isGlobalMuon1 = true;
      if (iMuon1->isTrackerMuon())    isTrackerMuon1 = true;
      if (iMuon1->isStandAloneMuon()) isStandAloneMuon1 = true;

      //cuts on muon track
      TrackRef inTrackMu1;
      inTrackMu1 = iMuon1->track();
      if( inTrackMu1.isNull())  continue;
      if(!(inTrackMu1->quality(reco::TrackBase::highPurity)))  continue;

      for (std::vector<pat::PackedCandidate>::const_iterator iTrack1 = thePATTrackHandle->begin(); iTrack1 != thePATTrackHandle->end(); ++iTrack1){

	//if (IsTheSame(*iTrack1,*iTrigMuon) ) continue;
        //if (IsTheSame(*iTrack1,*iMuon1) ) continue;

	//Nota bene: if you want to use dxy or dz you need to be sure 
	//the pt of the tracks is bigger than 0.5 GeV, otherwise you 
	//will get an error related to covariance matrix.
	//Next lines are very recommended

	if(iTrack1->pt() <= pi_pt_cut_) continue;
	if(std::abs(iTrack1->eta()) > pi_eta_cut_) continue;
	if(iTrack1->charge()==0) continue;// NO neutral objects
	if(std::abs(iTrack1->pdgId())!=211) continue;//Due to the lack of the particle ID all the tracks for cms are pions(ID==211)
	if(!(iTrack1->trackHighPurity())) continue; 


	bool hnl_pi_match = false;
	bool isMCMatchedTrack1 = false;

	if(run==1){
	  int match_pi_idx = getMatchedGenPartIdx(iTrack1->pt(),iTrack1->eta(),iTrack1->phi(),211,*packedGenParticleCollection);

	  if (match_pi_idx>0){
	    isMCMatchedTrack1 = true;
	    pat::PackedGenParticle matchedGenPi = (*packedGenParticleCollection).at(match_pi_idx);
	    if (matchedGenPi.motherRef().isNonnull() &&
		matchedGenPi.motherRef().isAvailable() &&
		std::abs(matchedGenPi.mother(0)->pdgId()) == 9900015 ){
	      hnl_pi_match=true;
	    }
	  }
	}

	//cuts on mupi mass and pt
	TLorentzVector p4mu1,p4pi1;
	p4pi1.SetPtEtaPhiM(iTrack1->pt(),iTrack1->eta(),iTrack1->phi(), pdg.PDG_PION_MASS);
	p4mu1.SetPtEtaPhiM(iMuon1->pt(), iMuon1->eta(), iMuon1->phi(), pdg.PDG_MUON_MASS);

	if ((p4mu1 + p4mu2 + p4pi1).M() > mumupi_mass_cut_) continue;
	if ((p4mu1 + p4pi1).M() >  mupi_mass_high_cut_) continue;
	if ((p4mu1 + p4pi1).M() <  mupi_mass_low_cut_) continue;
	if ((p4mu1 + p4pi1).Pt() <  mupi_pt_cut_) continue;


	TransientTrack muon1TT((*TTrackBuilder).build(inTrackMu1));
	TransientTrack pion1TT((*TTrackBuilder).build(iTrack1->pseudoTrack()));


	if(!muon1TT.isValid()) continue;
	if(!pion1TT.isValid()) continue;
	if(muon1TT == pion1TT) continue;


	float muon_sigma = pdg.PDG_MUON_MASS * 1.e-6;
	float chi = 0.;
	float ndf = 0.;

	vector < RefCountedKinematicParticle > hnlParticles;
	hnlParticles.push_back(pFactory.particle(muon1TT, pdg.PM_PDG_MUON_MASS, chi, ndf, muon_sigma));
	hnlParticles.push_back(pFactory.particle(pion1TT, pdg.PM_PDG_PION_MASS, chi, ndf, muon_sigma));
	KinematicParticleVertexFitter hnlToPiMu_vertexFitter;
	RefCountedKinematicTree hnlToPiMu_kinTree;
	hnlToPiMu_kinTree = hnlToPiMu_vertexFitter.fit(hnlParticles);

	if (!hnlToPiMu_kinTree->isValid()) continue;

	hnlToPiMu_kinTree->movePointerToTheTop();
	RefCountedKinematicParticle muPi_particle = hnlToPiMu_kinTree->currentParticle();
	RefCountedKinematicVertex   muPi_vtx      = hnlToPiMu_kinTree->currentDecayVertex();


	double muPi_mass = muPi_particle->currentState().mass();

	//if ( muPi_mass >  mupi_mass_high_cut_) continue;


	double muPi_vtxprob = TMath::Prob(muPi_vtx->chiSquared(), muPi_vtx->degreesOfFreedom());
	if(muPi_vtxprob < vtx_prob_cut_) continue;


	bool hnl_mu_match = false;
	bool isMCMatchedMuon1 = false;

	if(run==1){
	  int match_mu1_idx = getMatchedGenPartIdx(iMuon1->pt(),iMuon1->eta(),iMuon1->phi(),13,*packedGenParticleCollection);

	  if (match_mu1_idx>0){
	    isMCMatchedMuon1 = true;
	    pat::PackedGenParticle matchedGenMuon = (*packedGenParticleCollection).at(match_mu1_idx);
	    if (matchedGenMuon.motherRef().isNonnull() &&
		matchedGenMuon.motherRef().isAvailable() &&
		std::abs(matchedGenMuon.mother(0)->pdgId()) == 9900015 ){
	      hnl_mu_match=true;
	    }
	  }
	}

	reco::Vertex refitted_vertex_best = thePrimaryV;

	//Do not perfrorm vertex refit
	//at the moment to help speed up the production
	//Maybe it is not even needed


	/*
	   Double_t refittedVertex_x = -9999.; 
	   Double_t refittedVertex_y = -9999.; 
	   Double_t refittedVertex_z = -9999.; 
	   Double_t refittedVertex_xErr = -9999.; 
	   Double_t refittedVertex_yErr = -9999.; 
	   Double_t refittedVertex_zErr = -9999.; 
	   Double_t refittedVertex_prob = -9999.; 
	   Double_t refittedVertex_dN = 0.; 
	   Double_t cos3D_best = -9999. ; 
	   Double_t cos2D_best = -9999. ; 

	   for (size_t i = 0; i < recVtxs->size(); ++i) {
	   const reco::Vertex &recoVertex = (*recVtxs)[i];
	   std::vector <reco::TransientTrack> vertexTransientTracks_refit;

	   for (std::vector<TrackBaseRef>::const_iterator iTrack = recoVertex.tracks_begin(); iTrack != recoVertex.tracks_end(); ++iTrack) {
	   TrackRef trackRef = iTrack->castTo<TrackRef>();
	   if (!(inTrackMu1==trackRef || iTrack1->bestTrack()==trackRef.get()))
	   {
	   TransientTrack tt(trackRef, &(*bFieldHandle));
	   vertexTransientTracks_refit.push_back(tt);
	   }
	   }

	   reco::Vertex refitted_vertex = recoVertex;

	   if (vertexTransientTracks_refit.size() > 0 && (recoVertex.tracksSize() != vertexTransientTracks_refit.size())) {
	   GlobalPoint vextex_glbPoint = GlobalPoint(recoVertex.x(), recoVertex.y(), recoVertex.z());
	   AdaptiveVertexFitter theFitter;
	   TransientVertex v = theFitter.vertex(vertexTransientTracks_refit, vextex_glbPoint);
	   if (v.isValid()) {
	   refitted_vertex = reco::Vertex(v);
	   }
	   }

	   Double_t dx = muPi_vtx->position().x() - refitted_vertex.x();
	   Double_t dy = muPi_vtx->position().y() - refitted_vertex.y();
	   Double_t dz = muPi_vtx->position().z() - refitted_vertex.z();
	   Double_t px = muPi_particle->currentState().globalMomentum().x();
	   Double_t py = muPi_particle->currentState().globalMomentum().y();
	   Double_t pz = muPi_particle->currentState().globalMomentum().z();
	   Double_t cos3D = (px*dx + py*dy + pz*dz)/(sqrt(dx*dx + dy*dy + dz*dz)*muPi_particle->currentState().globalMomentum().mag());
	   Double_t cos2D = (px*dx + py*dy)/(sqrt(dx*dx + dy*dy)*sqrt(px*px + py*py));

	   if (cos3D > cos3D_best) {
	   cos3D_best = cos3D;
	   cos2D_best = cos2D;
	   refittedVertex_x    = refitted_vertex.x();
	   refittedVertex_y    = refitted_vertex.y();
	   refittedVertex_z    = refitted_vertex.z();
	   refittedVertex_xErr = refitted_vertex.covariance(0, 0);
	   refittedVertex_yErr = refitted_vertex.covariance(1, 1);
	   refittedVertex_zErr = refitted_vertex.covariance(2, 2);
	   refittedVertex_prob = (TMath::Prob(refitted_vertex.chi2(), (int) refitted_vertex.ndof()));
	   refittedVertex_dN   = recoVertex.tracksSize() - vertexTransientTracks_refit.size();
	   refitted_vertex_best = refitted_vertex;
	   }
	   }
	   */

	//Vertex has not been refitted here!!
	Double_t refittedVertex_x    = refitted_vertex_best.x();
	Double_t refittedVertex_y    = refitted_vertex_best.y();
	Double_t refittedVertex_z    = refitted_vertex_best.z();
	Double_t refittedVertex_xErr = refitted_vertex_best.covariance(0, 0);
	Double_t refittedVertex_yErr = refitted_vertex_best.covariance(1, 1);
	Double_t refittedVertex_zErr = refitted_vertex_best.covariance(2, 2);
	Double_t refittedVertex_prob = (TMath::Prob(refitted_vertex_best.chi2(), (int) refitted_vertex_best.ndof()));

	//compute cos2D and cos3D wrt beam spot
	Double_t dx = muPi_vtx->position().x() - (*theBeamSpot).position().x();
	Double_t dy = muPi_vtx->position().y() - (*theBeamSpot).position().y();
	Double_t dz = muPi_vtx->position().z() - (*theBeamSpot).position().z();
	Double_t px = muPi_particle->currentState().globalMomentum().x();
	Double_t py = muPi_particle->currentState().globalMomentum().y();
	Double_t pz = muPi_particle->currentState().globalMomentum().z();
	Double_t cos3D_best = (px*dx + py*dy + pz*dz)/(sqrt(dx*dx + dy*dy + dz*dz)*muPi_particle->currentState().globalMomentum().mag());
	Double_t cos2D_best = (px*dx + py*dy)/(sqrt(dx*dx + dy*dy)*sqrt(px*px + py*py));

	Double_t vx    = muPi_vtx->position().x();
	Double_t vy    = muPi_vtx->position().y();
	Double_t vz    = muPi_vtx->position().z();
	Double_t vxErr = muPi_vtx->error().cxx();
	Double_t vyErr = muPi_vtx->error().cyy();
	Double_t vzErr = muPi_vtx->error().czz();

	//   SAVE
	C_Hnl_vertex_prob ->push_back(muPi_vtxprob);
	C_Hnl_mass ->push_back(muPi_mass);
	C_Hnl_preFit_mass ->push_back((p4mu1 + p4pi1).M());
	C_Hnl_px ->push_back(muPi_particle->currentState().globalMomentum().x());
	C_Hnl_py ->push_back(muPi_particle->currentState().globalMomentum().y());
	C_Hnl_pz ->push_back(muPi_particle->currentState().globalMomentum().z());
	C_Hnl_vertex_x   ->push_back(vx);
	C_Hnl_vertex_y   ->push_back(vy);
	C_Hnl_vertex_z   ->push_back(vz);
	C_Hnl_vertex_xErr->push_back(vxErr);
	C_Hnl_vertex_yErr->push_back(vyErr);
	C_Hnl_vertex_zErr->push_back(vzErr);
	C_Hnl_vertex_sig->push_back(sqrt(vx*vx + vy*vy + vz*vz)/sqrt(vxErr*vxErr + vyErr*vyErr + vzErr*vzErr));
	C_Hnl_vertex_cos3D->push_back(cos3D_best);
	C_Hnl_vertex_cos2D->push_back(cos2D_best);

	C_mu1_px ->push_back(p4mu1.Px());
	C_mu1_py ->push_back(p4mu1.Py());
	C_mu1_pz ->push_back(p4mu1.Pz());
	C_mu1_eta ->push_back(p4mu1.Eta());
	C_mu1_phi ->push_back(p4mu1.Phi());
	C_mu1_BS_ips_xy ->push_back(iMuon1->dB(pat::Muon::BS2D)/iMuon1->edB(pat::Muon::BS2D));
	C_mu1_BS_ips    ->push_back(iMuon1->dB(pat::Muon::BS3D)/iMuon1->edB(pat::Muon::BS3D));
	C_mu1_PV_ips_z  ->push_back(iMuon1->dB(pat::Muon::PVDZ)/iMuon1->edB(pat::Muon::PVDZ));
	C_mu1_BS_ip_xy  ->push_back(iMuon1->dB(pat::Muon::BS2D));
	C_mu1_BS_ip     ->push_back(iMuon1->dB(pat::Muon::BS3D));
	C_mu1_PV_ip_z   ->push_back(iMuon1->dB(pat::Muon::PVDZ));
	C_mu1_isHnlDaughter->push_back(hnl_mu_match ? 1 : 0);
	C_mu1_charge->push_back(iMuon1->charge());
	C_mu1_isSoft   ->push_back(isSoftMuon1? 1: 0);
	C_mu1_isLoose  ->push_back(isLooseMuon1? 1: 0);
	C_mu1_isMedium ->push_back(isMediumMuon1? 1: 0);
	C_mu1_isGlobal   ->push_back(isGlobalMuon1? 1: 0);
	C_mu1_isTracker  ->push_back(isTrackerMuon1? 1: 0);
	C_mu1_isStandAlone ->push_back(isStandAloneMuon1? 1: 0);
	C_mu1_isMCMatched ->push_back(isMCMatchedMuon1? 1: 0);


	C_trigMu_px ->push_back(p4mu2.Px());
	C_trigMu_py ->push_back(p4mu2.Py());
	C_trigMu_pz ->push_back(p4mu2.Pz());
	C_trigMu_eta ->push_back(p4mu2.Eta());
	C_trigMu_phi ->push_back(p4mu2.Phi());
	C_trigMu_BS_ips_xy ->push_back(iTrigMuon->dB(pat::Muon::BS2D)/iTrigMuon->edB(pat::Muon::BS2D));
	C_trigMu_BS_ips    ->push_back(iTrigMuon->dB(pat::Muon::BS3D)/iTrigMuon->edB(pat::Muon::BS3D));
	C_trigMu_PV_ips_z  ->push_back(iTrigMuon->dB(pat::Muon::PVDZ)/iTrigMuon->edB(pat::Muon::PVDZ));
	C_trigMu_BS_ip_xy  ->push_back(iTrigMuon->dB(pat::Muon::BS2D));
	C_trigMu_BS_ip     ->push_back(iTrigMuon->dB(pat::Muon::BS3D));
	C_trigMu_PV_ip_z   ->push_back(iTrigMuon->dB(pat::Muon::PVDZ));
	C_trigMu_charge ->push_back(iTrigMuon->charge());
	C_trigMu_isHnlBrother->push_back(is_hnl_brother ? 1 : 0);
	C_trigMu_isSoft   ->push_back(isSoftTrigMuon? 1: 0);
	C_trigMu_isLoose  ->push_back(isLooseTrigMuon? 1: 0);
	C_trigMu_isMedium ->push_back(isMediumTrigMuon? 1: 0);
	C_trigMu_isGlobal   ->push_back(isGlobalTrigMuon? 1: 0);
	C_trigMu_isTracker  ->push_back(isTrackerTrigMuon? 1: 0);
	C_trigMu_isStandAlone ->push_back(isStandAloneTrigMuon? 1: 0);
	C_trigMu_isMCMatched ->push_back(isMCMatchedTrigMuon? 1: 0);

	C_pi_charge ->push_back(iTrack1->charge());
	C_pi_px ->push_back(p4pi1.Px());
	C_pi_py ->push_back(p4pi1.Py());
	C_pi_pz ->push_back(p4pi1.Pz());
	C_pi_eta ->push_back(p4pi1.Eta());
	C_pi_phi ->push_back(p4pi1.Phi());
	C_pi_BS_ips_xy->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position()))/std::abs(iTrack1->dxyError()));
	C_pi_BS_ips_z ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position()))/std::abs(iTrack1->dzError()));
	C_pi_BS_ip_xy ->push_back(std::abs(iTrack1->dxy((*theBeamSpot).position())));
	C_pi_BS_ip_z  ->push_back(std::abs(iTrack1->dz ((*theBeamSpot).position())));
	C_pi_isMCMatched ->push_back(isMCMatchedTrack1? 1: 0);
	C_pi_isHnlDaughter->push_back(hnl_pi_match ? 1 : 0);

	C_mass->push_back((p4mu1 + p4mu2 + p4pi1).M());
	C_mu1mu2_mass->push_back((p4mu1 + p4mu2).M());
	C_mu1mu2_dr->push_back(deltaR(*iMuon1,*iTrigMuon));
	C_mu1pi_dr->push_back(deltaR(*iMuon1,*iTrack1));
	C_mu2pi_dr->push_back(deltaR(*iTrigMuon,*iTrack1));
	C_px ->push_back((p4mu1 + p4mu2 + p4pi1).Px());
	C_py ->push_back((p4mu1 + p4mu2 + p4pi1).Py());
	C_pz ->push_back((p4mu1 + p4mu2 + p4pi1).Pz());

	PV_x ->push_back(refittedVertex_x);
	PV_y ->push_back(refittedVertex_y);
	PV_z ->push_back(refittedVertex_z);
	PV_xErr ->push_back(refittedVertex_xErr);
	PV_yErr ->push_back(refittedVertex_yErr);
	PV_zErr ->push_back(refittedVertex_zErr);
	PV_prob ->push_back(refittedVertex_prob);
	//PV_dN ->push_back(refittedVertex_dN);

	mu7_ip4_matched->push_back(TriggerMatches[0]);
	mu8_ip3_matched->push_back(TriggerMatches[1]);
	mu8_ip3p5_matched ->push_back(TriggerMatches[2]);
	mu8_ip5_matched->push_back(TriggerMatches[3]);
	mu8_ip6_matched->push_back(TriggerMatches[4]);
	mu9_ip4_matched->push_back(TriggerMatches[5]);
	mu9_ip5_matched->push_back(TriggerMatches[6]);
	mu9_ip6_matched->push_back(TriggerMatches[7]);
	mu10p5_ip3p5_matched->push_back(TriggerMatches[8]);
	mu12_ip6_matched->push_back(TriggerMatches[9]);

	mu7_ip4_eta->push_back(TriggerPathEta[0]);
	mu8_ip3_eta->push_back(TriggerPathEta[1]);
	mu8_ip3p5_eta ->push_back(TriggerPathEta[2]);
	mu8_ip5_eta->push_back(TriggerPathEta[3]);
	mu8_ip6_eta->push_back(TriggerPathEta[4]);
	mu9_ip4_eta->push_back(TriggerPathEta[5]);
	mu9_ip5_eta->push_back(TriggerPathEta[6]);
	mu9_ip6_eta->push_back(TriggerPathEta[7]);
	mu10p5_ip3p5_eta->push_back(TriggerPathEta[8]);
	mu12_ip6_eta->push_back(TriggerPathEta[9]);

	mu7_ip4_pt->push_back(TriggerPathPt[0]);
	mu8_ip3_pt->push_back(TriggerPathPt[1]);
	mu8_ip3p5_pt ->push_back(TriggerPathPt[2]);
	mu8_ip5_pt->push_back(TriggerPathPt[3]);
	mu8_ip6_pt->push_back(TriggerPathPt[4]);
	mu9_ip4_pt->push_back(TriggerPathPt[5]);
	mu9_ip5_pt->push_back(TriggerPathPt[6]);
	mu9_ip6_pt->push_back(TriggerPathPt[7]);
	mu10p5_ip3p5_pt->push_back(TriggerPathPt[8]);
	mu12_ip6_pt->push_back(TriggerPathPt[9]);

	mu7_ip4_dr->push_back(TriggerPathDR[0]);
	mu8_ip3_dr->push_back(TriggerPathDR[1]);
	mu8_ip3p5_dr ->push_back(TriggerPathDR[2]);
	mu8_ip5_dr->push_back(TriggerPathDR[3]);
	mu8_ip6_dr->push_back(TriggerPathDR[4]);
	mu9_ip4_dr->push_back(TriggerPathDR[5]);
	mu9_ip5_dr->push_back(TriggerPathDR[6]);
	mu9_ip6_dr->push_back(TriggerPathDR[7]);
	mu10p5_ip3p5_dr->push_back(TriggerPathDR[8]);
	mu12_ip6_dr->push_back(TriggerPathDR[9]);

	mu7_ip4_fired->push_back(TriggersFired[0]);
	mu8_ip3_fired->push_back(TriggersFired[1]);
	mu8_ip3p5_fired ->push_back(TriggersFired[2]);
	mu8_ip5_fired->push_back(TriggersFired[3]);
	mu8_ip6_fired->push_back(TriggersFired[4]);
	mu9_ip4_fired->push_back(TriggersFired[5]);
	mu9_ip5_fired->push_back(TriggersFired[6]);
	mu9_ip6_fired->push_back(TriggersFired[7]);
	mu10p5_ip3p5_fired->push_back(TriggersFired[8]);
	mu12_ip6_fired->push_back(TriggersFired[9]);

	++nCand;

	hnlParticles.clear();
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

  C_Hnl_vertex_prob->clear();
  C_Hnl_mass->clear(); 
  C_Hnl_preFit_mass->clear(); 
  C_Hnl_px->clear();
  C_Hnl_py->clear();
  C_Hnl_pz->clear();
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
  C_mu1_px->clear();
  C_mu1_py->clear();
  C_mu1_pz->clear();
  C_mu1_eta->clear();
  C_mu1_phi->clear();
  C_mu1_BS_ips_xy->clear();
  C_mu1_BS_ips->clear();
  C_mu1_PV_ips_z->clear();
  C_mu1_BS_ip_xy->clear();
  C_mu1_BS_ip->clear();
  C_mu1_PV_ip_z->clear();
  C_mu1_charge->clear();
  C_mu1_isSoft->clear();
  C_mu1_isLoose->clear();
  C_mu1_isMedium->clear();
  C_mu1_isGlobal->clear();
  C_mu1_isTracker->clear();
  C_mu1_isStandAlone->clear();
  C_mu1_isMCMatched->clear();

  C_trigMu_px->clear();
  C_trigMu_py->clear();
  C_trigMu_pz->clear();
  C_trigMu_eta->clear();
  C_trigMu_phi->clear();
  C_trigMu_BS_ips_xy->clear();
  C_trigMu_BS_ips->clear();
  C_trigMu_PV_ips_z->clear();
  C_trigMu_BS_ip_xy->clear();
  C_trigMu_BS_ip->clear();
  C_trigMu_PV_ip_z->clear();
  C_trigMu_charge->clear();
  C_trigMu_isSoft->clear();
  C_trigMu_isLoose->clear();
  C_trigMu_isMedium->clear();
  C_trigMu_isGlobal->clear();
  C_trigMu_isTracker->clear();
  C_trigMu_isStandAlone->clear();
  C_trigMu_isMCMatched->clear();
  //
  C_mass->clear();
  C_mu1mu2_mass->clear();
  C_mu1mu2_dr->clear();
  C_mu1pi_dr->clear();
  C_mu2pi_dr->clear();
  C_pi_charge->clear();
  C_px->clear();
  C_py->clear();
  C_pz->clear();
  C_pi_px->clear(); 
  C_pi_py->clear();
  C_pi_pz->clear();
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

  mu7_ip4_matched->clear();
  mu8_ip3_matched->clear();
  mu8_ip3p5_matched ->clear();
  mu8_ip5_matched->clear();
  mu8_ip6_matched->clear();
  mu9_ip4_matched->clear();
  mu9_ip5_matched->clear();
  mu9_ip6_matched->clear();
  mu10p5_ip3p5_matched->clear();
  mu12_ip6_matched->clear();

  mu7_ip4_eta->clear();
  mu8_ip3_eta->clear();
  mu8_ip3p5_eta ->clear();
  mu8_ip5_eta->clear();
  mu8_ip6_eta->clear();
  mu9_ip4_eta->clear();
  mu9_ip5_eta->clear();
  mu9_ip6_eta->clear();
  mu10p5_ip3p5_eta->clear();
  mu12_ip6_eta->clear();

  mu7_ip4_pt->clear();
  mu8_ip3_pt->clear();
  mu8_ip3p5_pt ->clear();
  mu8_ip5_pt->clear();
  mu8_ip6_pt->clear();
  mu9_ip4_pt->clear();
  mu9_ip5_pt->clear();
  mu9_ip6_pt->clear();
  mu10p5_ip3p5_pt->clear();
  mu12_ip6_pt->clear();

  mu7_ip4_dr->clear();
  mu8_ip3_dr->clear();
  mu8_ip3p5_dr ->clear();
  mu8_ip5_dr->clear();
  mu8_ip6_dr->clear();
  mu9_ip4_dr->clear();
  mu9_ip5_dr->clear();
  mu9_ip6_dr->clear();
  mu10p5_ip3p5_dr->clear();
  mu12_ip6_dr->clear();

  mu7_ip4_fired->clear();
  mu8_ip3_fired->clear();
  mu8_ip3p5_fired ->clear();
  mu8_ip5_fired->clear();
  mu8_ip6_fired->clear();
  mu9_ip4_fired->clear();
  mu9_ip5_fired->clear();
  mu9_ip6_fired->clear();
  mu10p5_ip3p5_fired->clear();
  mu12_ip6_fired->clear();

  C_pi_isHnlDaughter->clear();
  C_mu1_isHnlDaughter->clear();
  C_trigMu_isHnlBrother->clear();


}

template <typename A,typename B> bool hnlAnalyzer_miniAOD::IsTheSame(const A& cand1, const B& cand2){
  double deltaPt  = std::abs(cand1.pt()-cand2.pt());
  double deltaEta = cand1.eta()-cand2.eta();

  auto deltaPhi = std::abs(cand1.phi() - cand2.phi());
  if (deltaPhi > float(M_PI))
    deltaPhi -= float(2 * M_PI);
  double deltaR2 = deltaEta*deltaEta + deltaPhi*deltaPhi;
  if(deltaR2<0.01 && deltaPt<0.1) return true;
  else return false;
}

int hnlAnalyzer_miniAOD::getMatchedGenPartIdx(double pt, double eta, double phi, int pdg_id, std::vector<pat::PackedGenParticle> packedGen_particles){
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
void hnlAnalyzer_miniAOD::beginJob()
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

  wwtree->Branch("C_Hnl_vertex_prob", &C_Hnl_vertex_prob);
  wwtree->Branch("C_Hnl_mass", &C_Hnl_mass);
  wwtree->Branch("C_Hnl_preFit_mass", &C_Hnl_preFit_mass);
  wwtree->Branch("C_Hnl_px", &C_Hnl_px);
  wwtree->Branch("C_Hnl_py", &C_Hnl_py);
  wwtree->Branch("C_Hnl_pz", &C_Hnl_pz);
  wwtree->Branch("C_Hnl_vertex_x" , &C_Hnl_vertex_x);
  wwtree->Branch("C_Hnl_vertex_y" , &C_Hnl_vertex_y);
  wwtree->Branch("C_Hnl_vertex_z" , &C_Hnl_vertex_z);
  wwtree->Branch("C_Hnl_vertex_xErr" , &C_Hnl_vertex_xErr);
  wwtree->Branch("C_Hnl_vertex_yErr" , &C_Hnl_vertex_yErr);
  wwtree->Branch("C_Hnl_vertex_zErr" , &C_Hnl_vertex_zErr);
  wwtree->Branch("C_Hnl_vertex_sig" , &C_Hnl_vertex_sig);
  wwtree->Branch("C_Hnl_vertex_cos3D" , &C_Hnl_vertex_cos3D);
  wwtree->Branch("C_Hnl_vertex_cos2D" , &C_Hnl_vertex_cos2D);

  wwtree->Branch("C_mu1_px"    , &C_mu1_px);
  wwtree->Branch("C_mu1_py"    , &C_mu1_py);
  wwtree->Branch("C_mu1_pz"    , &C_mu1_pz);
  wwtree->Branch("C_mu1_eta"   , &C_mu1_eta);
  wwtree->Branch("C_mu1_phi"   , &C_mu1_phi);
  wwtree->Branch("C_mu1_BS_ips_xy", &C_mu1_BS_ips_xy);
  wwtree->Branch("C_mu1_BS_ips", &C_mu1_BS_ips);
  wwtree->Branch("C_mu1_PV_ips_z" , &C_mu1_PV_ips_z);
  wwtree->Branch("C_mu1_BS_ip_xy" , &C_mu1_BS_ip_xy);
  wwtree->Branch("C_mu1_BS_ip" , &C_mu1_BS_ip);
  wwtree->Branch("C_mu1_PV_ip_z"  , &C_mu1_PV_ip_z);
  wwtree->Branch("C_mu1_charge", &C_mu1_charge);
  wwtree->Branch("C_mu1_isSoft", &C_mu1_isSoft);
  wwtree->Branch("C_mu1_isLoose", &C_mu1_isLoose);
  wwtree->Branch("C_mu1_isMedium", &C_mu1_isMedium);
  wwtree->Branch("C_mu1_isGlobal", &C_mu1_isGlobal);
  wwtree->Branch("C_mu1_isTracker", &C_mu1_isTracker);
  wwtree->Branch("C_mu1_isStandAlone", &C_mu1_isStandAlone);
  wwtree->Branch("C_mu1_isMCMatched", &C_mu1_isMCMatched);

  wwtree->Branch("C_trigMu_px"    , &C_trigMu_px);
  wwtree->Branch("C_trigMu_py"    , &C_trigMu_py);
  wwtree->Branch("C_trigMu_pz"    , &C_trigMu_pz);
  wwtree->Branch("C_trigMu_eta"   , &C_trigMu_eta);
  wwtree->Branch("C_trigMu_phi"   , &C_trigMu_phi);
  wwtree->Branch("C_trigMu_BS_ips_xy", &C_trigMu_BS_ips_xy);
  wwtree->Branch("C_trigMu_BS_ips", &C_trigMu_BS_ips);
  wwtree->Branch("C_trigMu_PV_ips_z" , &C_trigMu_PV_ips_z);
  wwtree->Branch("C_trigMu_BS_ip_xy" , &C_trigMu_BS_ip_xy);
  wwtree->Branch("C_trigMu_BS_ip" , &C_trigMu_BS_ip);
  wwtree->Branch("C_trigMu_PV_ip_z"  , &C_trigMu_PV_ip_z);
  wwtree->Branch("C_trigMu_charge", &C_trigMu_charge);
  wwtree->Branch("C_trigMu_isSoft", &C_trigMu_isSoft);
  wwtree->Branch("C_trigMu_isLoose", &C_trigMu_isLoose);
  wwtree->Branch("C_trigMu_isMedium", &C_trigMu_isMedium);
  wwtree->Branch("C_trigMu_isGlobal", &C_trigMu_isGlobal);
  wwtree->Branch("C_trigMu_isTracker", &C_trigMu_isTracker);
  wwtree->Branch("C_trigMu_isStandAlone", &C_trigMu_isStandAlone);
  wwtree->Branch("C_trigMu_isMCMatched", &C_trigMu_isMCMatched);

  wwtree->Branch("C_mass"            , &C_mass          );
  wwtree->Branch("C_mu1mu2_mass"     , &C_mu1mu2_mass   );
  wwtree->Branch("C_mu1mu2_dr"       , &C_mu1mu2_dr   );
  wwtree->Branch("C_mu1pi_dr"        , &C_mu1pi_dr   );
  wwtree->Branch("C_mu2pi_dr"        , &C_mu2pi_dr   );
  wwtree->Branch("C_pi_charge"       , &C_pi_charge     );
  wwtree->Branch("C_px"              , &C_px            );
  wwtree->Branch("C_py"              , &C_py            );
  wwtree->Branch("C_pz"              , &C_pz            );
  wwtree->Branch("C_pi_px"           , &C_pi_px         );
  wwtree->Branch("C_pi_py"           , &C_pi_py         );
  wwtree->Branch("C_pi_pz"           , &C_pi_pz         );
  wwtree->Branch("C_pi_eta"          , &C_pi_eta        );
  wwtree->Branch("C_pi_phi"          , &C_pi_phi        );
  wwtree->Branch("C_pi_BS_ips_xy"       , &C_pi_BS_ips_xy     );
  wwtree->Branch("C_pi_BS_ips_z"        , &C_pi_BS_ips_z     );
  wwtree->Branch("C_pi_BS_ip_xy"        , &C_pi_BS_ip_xy     );
  wwtree->Branch("C_pi_BS_ip_z"         , &C_pi_BS_ip_z     );
  wwtree->Branch("C_pi_isMCMatched", &C_pi_isMCMatched);

  wwtree->Branch("PV_x"    , &PV_x);
  wwtree->Branch("PV_y"    , &PV_y);
  wwtree->Branch("PV_z"    , &PV_z);
  wwtree->Branch("PV_xErr" , &PV_xErr);
  wwtree->Branch("PV_yErr" , &PV_yErr);
  wwtree->Branch("PV_zErr" , &PV_zErr);
  wwtree->Branch("PV_prob" , &PV_prob);
  //wwtree->Branch("PV_dN"   , &PV_dN);

  wwtree->Branch("mu7_ip4_matched",    &mu7_ip4_matched);
  wwtree->Branch("mu8_ip3_matched",    &mu8_ip3_matched);
  wwtree->Branch("mu8_ip3p5_matched",  &mu8_ip3p5_matched);
  wwtree->Branch("mu8_ip5_matched",    &mu8_ip5_matched);
  wwtree->Branch("mu8_ip6_matched",    &mu8_ip6_matched);
  wwtree->Branch("mu9_ip4_matched",    &mu9_ip4_matched);
  wwtree->Branch("mu9_ip5_matched",    &mu9_ip5_matched);
  wwtree->Branch("mu9_ip6_matched",    &mu9_ip6_matched);
  wwtree->Branch("mu10p5_ip3p5_matched", &mu10p5_ip3p5_matched);
  wwtree->Branch("mu12_ip6_matched",   &mu12_ip6_matched);

  wwtree->Branch("mu7_ip4_eta",    &mu7_ip4_eta);
  wwtree->Branch("mu8_ip3_eta",    &mu8_ip3_eta);
  wwtree->Branch("mu8_ip3p5_eta",  &mu8_ip3p5_eta);
  wwtree->Branch("mu8_ip5_eta",    &mu8_ip5_eta);
  wwtree->Branch("mu8_ip6_eta",    &mu8_ip6_eta);
  wwtree->Branch("mu9_ip4_eta",    &mu9_ip4_eta);
  wwtree->Branch("mu9_ip5_eta",    &mu9_ip5_eta);
  wwtree->Branch("mu9_ip6_eta",    &mu9_ip6_eta);
  wwtree->Branch("mu10p5_ip3p5_eta", &mu10p5_ip3p5_eta);
  wwtree->Branch("mu12_ip6_eta",   &mu12_ip6_eta);

  wwtree->Branch("mu7_ip4_pt",    &mu7_ip4_pt);
  wwtree->Branch("mu8_ip3_pt",    &mu8_ip3_pt);
  wwtree->Branch("mu8_ip3p5_pt",  &mu8_ip3p5_pt);
  wwtree->Branch("mu8_ip5_pt",    &mu8_ip5_pt);
  wwtree->Branch("mu8_ip6_pt",    &mu8_ip6_pt);
  wwtree->Branch("mu9_ip4_pt",    &mu9_ip4_pt);
  wwtree->Branch("mu9_ip5_pt",    &mu9_ip5_pt);
  wwtree->Branch("mu9_ip6_pt",    &mu9_ip6_pt);
  wwtree->Branch("mu10p5_ip3p5_pt", &mu10p5_ip3p5_pt);
  wwtree->Branch("mu12_ip6_pt",   &mu12_ip6_pt);

  wwtree->Branch("mu7_ip4_dr",    &mu7_ip4_dr);
  wwtree->Branch("mu8_ip3_dr",    &mu8_ip3_dr);
  wwtree->Branch("mu8_ip3p5_dr",  &mu8_ip3p5_dr);
  wwtree->Branch("mu8_ip5_dr",    &mu8_ip5_dr);
  wwtree->Branch("mu8_ip6_dr",    &mu8_ip6_dr);
  wwtree->Branch("mu9_ip4_dr",    &mu9_ip4_dr);
  wwtree->Branch("mu9_ip5_dr",    &mu9_ip5_dr);
  wwtree->Branch("mu9_ip6_dr",    &mu9_ip6_dr);
  wwtree->Branch("mu10p5_ip3p5_dr", &mu10p5_ip3p5_dr);
  wwtree->Branch("mu12_ip6_dr",   &mu12_ip6_dr);

  wwtree->Branch("mu7_ip4_fired",    &mu7_ip4_fired);
  wwtree->Branch("mu8_ip3_fired",    &mu8_ip3_fired);
  wwtree->Branch("mu8_ip3p5_fired",  &mu8_ip3p5_fired);
  wwtree->Branch("mu8_ip5_fired",    &mu8_ip5_fired);
  wwtree->Branch("mu8_ip6_fired",    &mu8_ip6_fired);
  wwtree->Branch("mu9_ip4_fired",    &mu9_ip4_fired);
  wwtree->Branch("mu9_ip5_fired",    &mu9_ip5_fired);
  wwtree->Branch("mu9_ip6_fired",    &mu9_ip6_fired);
  wwtree->Branch("mu10p5_ip3p5_fired", &mu10p5_ip3p5_fired);
  wwtree->Branch("mu12_ip6_fired",   &mu12_ip6_fired);

  wwtree->Branch("C_pi_isHnlDaughter"        , &C_pi_isHnlDaughter        );
  wwtree->Branch("C_mu1_isHnlDaughter"        , &C_mu1_isHnlDaughter        );
  wwtree->Branch("C_trigMu_isHnlBrother"        , &C_trigMu_isHnlBrother        );

}

// ------------ method called once each job just after ending the event loop  ------------
  void
hnlAnalyzer_miniAOD::endJob()
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
hnlAnalyzer_miniAOD::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(hnlAnalyzer_miniAOD);
