import ROOT
import sys
import math
import os
import json

configFileName = sys.argv[1]
dataset_category = sys.argv[2]

with open(configFileName, "r") as f:
    config = json.loads(f.read())

#get input files
inputFileName_list = config[dataset_category]

chain = ROOT.TChain('wztree')
for inputFileName in inputFileName_list:
    chain.Add(inputFileName)

input_file_name = inputFileName_list[0].split("/")[-1].split(".")[0]
dataset_name_label = input_file_name[input_file_name.find("_")+1:input_file_name.find("_tree")]

outputFileName = "out_hnl_tree_analyzer_"+dataset_name_label+".root"
outputDirName = "/afs/cern.ch/work/l/llunerti/private/CMSSW_10_2_27/src/hnlAnalysis/analyzer"

outputFile = ROOT.TFile(os.path.join(outputDirName,outputFileName),"RECREATE")

m_hnl_pt           = ROOT.RDF.TH1DModel("h_hnl_pt"          ,";HNL p_{T} [GeV];Events", 10, 0., 10.)
m_hnl_preFit_mass  = ROOT.RDF.TH1DModel("h_hnl_preFit_mass" ,";HNL (pre-fit) mass [GeV];Events", 20, 0.25, 6.)
m_hnl_postFit_mass = ROOT.RDF.TH1DModel("h_hnl_postFit_mass",";HNL (post-fit) mass [GeV];Events", 20, 0.25, 6.)
m_hnl_postFit_mass_ss = ROOT.RDF.TH1DModel("h_hnl_postFit_mass_ss",";HNL (post-fit) mass [GeV];Events", 20, 0.25, 6.)
m_hnl_lxy          = ROOT.RDF.TH1DModel("h_hnl_lxy"         ,";HNL L_{xy} [cm];Events", 20, 0., 40.)
m_hnl_vtx_prob     = ROOT.RDF.TH1DModel("h_hnl_vtx_prob"    ,";SV probability;Events", 100, 0., 1.)
m_hnl_vtx_cos2D    = ROOT.RDF.TH1DModel("h_hnl_vtx_cos2D"   ,";SV cos2D;Events", 100, -1., 1.)
m_hnl_vtx_dispSig  = ROOT.RDF.TH1DModel("h_hnl_vtx_dispSig" ,";SV displacement significance;Events", 100, 0., 200.)

m_mu1_pt           = ROOT.RDF.TH1DModel("h_mu1_pt"    ,";#mu_{1} p_{T} [GeV];Events", 10, 0., 10.)
m_mu1_eta          = ROOT.RDF.TH1DModel("h_mu1_eta"   ,";#mu_{1} #eta;Events", 20, -2.5, 2.5)
m_mu1_ip_z         = ROOT.RDF.TH1DModel("h_mu1_ip_z"  ,";#mu_{1} z IP [cm];Events", 20, 0., 5.)
m_mu1_ip_xy        = ROOT.RDF.TH1DModel("h_mu1_ip_xy" ,";#mu_{1} xy IP [cm];Events", 20, 0., 5.)
m_mu1_ips_z        = ROOT.RDF.TH1DModel("h_mu1_ips_z" ,";#mu_{1} z IPS;Events", 20, 0., 200.)
m_mu1_ips_xy       = ROOT.RDF.TH1DModel("h_mu1_ips_xy",";#mu_{1} xy IPS;Events", 20, 0., 200.)

m_mu2_pt           = ROOT.RDF.TH1DModel("h_mu2_pt"  ,";#mu_{2} p_{T} [GeV];Events", 10, 0., 10.)
m_mu2_eta          = ROOT.RDF.TH1DModel("h_mu2_eta" ,";#mu_{2} #eta;Events", 20, -2.5, 2.5)

m_pi_pt            = ROOT.RDF.TH1DModel("h_pi_pt"    ,";#pi p_{T} [GeV];Events", 10, 0., 10.)
m_pi_eta           = ROOT.RDF.TH1DModel("h_pi_eta"   ,";#pi #eta;Events", 20, -2.5, 2.5)
m_pi_ip_z          = ROOT.RDF.TH1DModel("h_pi_ip_z"  ,";#pi z IP [cm];Events", 20, 0., 5.)
m_pi_ip_xy         = ROOT.RDF.TH1DModel("h_pi_ip_xy" ,";#pi xy IP [cm];Events", 20, 0., 5.)
m_pi_ips_z         = ROOT.RDF.TH1DModel("h_pi_ips_z" ,";#pi z IPS;Events", 20, 0., 200.)
m_pi_ips_xy        = ROOT.RDF.TH1DModel("h_pi_ips_xy",";#pi xy IPS;Events", 20, 0., 200.)

df = ROOT.RDataFrame(chain)
df = df.Define("C_Hnl_pt" ,"sqrt(C_Hnl_px*C_Hnl_px + C_Hnl_py*C_Hnl_py)")\
       .Define("C_Hnl_lxy","sqrt(C_Hnl_vertex_x*C_Hnl_vertex_x + C_Hnl_vertex_y*C_Hnl_vertex_y)")\
       .Define("C_mu1_pt" ,"sqrt(C_mu1_px*C_mu1_px + C_mu1_py*C_mu1_py)")\
       .Define("C_mu2_pt" ,"sqrt(C_mu2_px*C_mu2_px + C_mu2_py*C_mu2_py)")\
       .Define("C_pi_pt"  ,"sqrt(C_pi_px*C_pi_px + C_pi_py*C_pi_py)")\
       .Define("mask","ArgMax(C_Hnl_pt)")

#keep best hnl pt combo from each event
for c in df.GetColumnNames():
    col_name = str(c)
    col_type = df.GetColumnType(col_name)
    if col_type.find("ROOT::VecOps")<0:
        continue
    df = df.Define(col_name+"_ptBest",col_name+"[mask]")

#get mc truth in case of signal sample
if dataset_category == "signal":
    df = df.Filter("C_mu1_isHnlDaughter_ptBest>0 && C_pi_isHnlDaughter_ptBest>0 && C_mu2_isHnlBrother_ptBest>0")

h_hnl_pt           = df.Histo1D(m_hnl_pt          ,"C_Hnl_pt_ptBest")
h_hnl_preFit_mass  = df.Histo1D(m_hnl_preFit_mass ,"C_Hnl_preFit_mass_ptBest")
h_hnl_postFit_mass = df.Histo1D(m_hnl_postFit_mass,"C_Hnl_mass_ptBest")
h_hnl_postFit_mass_ss = df.Filter("C_mu1_charge_ptBest+C_pi_charge_ptBest != 0").Histo1D(m_hnl_postFit_mass_ss,"C_Hnl_mass_ptBest")
h_hnl_lxy          = df.Histo1D(m_hnl_lxy         ,"C_Hnl_lxy_ptBest")
h_hnl_vtx_prob     = df.Histo1D(m_hnl_vtx_prob    ,"C_Hnl_vertex_prob_ptBest")
h_hnl_vtx_cos2D    = df.Histo1D(m_hnl_vtx_cos2D   ,"C_Hnl_vertex_cos2D_ptBest")
h_hnl_vtx_dispSig  = df.Histo1D(m_hnl_vtx_dispSig ,"C_Hnl_vertex_sig_ptBest")
                                                  
h_mu2_pt           = df.Histo1D(m_mu2_pt       ,"C_mu2_pt_ptBest")
h_mu2_eta          = df.Histo1D(m_mu2_eta      ,"C_mu2_eta_ptBest")
                                      
h_mu1_pt           = df.Histo1D(m_mu1_pt     ,"C_mu1_pt_ptBest")
h_mu1_eta          = df.Histo1D(m_mu1_eta    ,"C_mu1_eta_ptBest")
h_mu1_ip_z         = df.Histo1D(m_mu1_ip_z   ,"C_mu1_ip_z_ptBest")
h_mu1_ip_xy        = df.Histo1D(m_mu1_ip_xy  ,"C_mu1_ip_xy_ptBest")
h_mu1_ips_z        = df.Histo1D(m_mu1_ips_z  ,"C_mu1_ips_z_ptBest")
h_mu1_ips_xy       = df.Histo1D(m_mu1_ips_xy ,"C_mu1_ips_xy_ptBest")
                                             
h_pi_pt            = df.Histo1D(m_pi_pt      ,"C_pi_pt_ptBest")
h_pi_eta           = df.Histo1D(m_pi_eta     ,"C_pi_eta_ptBest")
h_pi_ip_z          = df.Histo1D(m_pi_ip_z    ,"C_pi_ip_z_ptBest")
h_pi_ip_xy         = df.Histo1D(m_pi_ip_xy   ,"C_pi_ip_xy_ptBest")
h_pi_ips_z         = df.Histo1D(m_pi_ips_z   ,"C_pi_ips_z_ptBest")
h_pi_ips_xy        = df.Histo1D(m_pi_ips_xy  ,"C_pi_ips_xy_ptBest")

h_hnl_pt          .Write()
h_hnl_preFit_mass .Write()
h_hnl_postFit_mass.Write()
h_hnl_postFit_mass_ss.Write()
h_hnl_lxy         .Write()
h_hnl_vtx_prob    .Write()
h_hnl_vtx_cos2D   .Write()
h_hnl_vtx_dispSig .Write()
h_mu2_pt          .Write()
h_mu2_eta         .Write()
h_mu1_pt          .Write()
h_mu1_eta         .Write()
h_mu1_ip_z        .Write()
h_mu1_ip_xy       .Write()
h_mu1_ips_z       .Write()
h_mu1_ips_xy      .Write()
h_pi_pt           .Write()
h_pi_eta          .Write()
h_pi_ip_z         .Write()
h_pi_ip_xy        .Write()
h_pi_ips_z        .Write()
h_pi_ips_xy       .Write()

outputFile.Close()
