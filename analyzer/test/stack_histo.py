import subprocess
import sys
import os
import json
script, configFileName = sys.argv

with open(configFileName, "r") as f:
    config = json.loads(f.read())
import ROOT

inputDirName = str(config["inputDirName"])
outDirName = str(config["outDirName"])

for plotName in config["plotNameList"]:
    ROOT.gROOT.SetBatch(ROOT.kTRUE)

    c1 = ROOT.TCanvas("c1","c1",800,800)
    c2 = ROOT.TCanvas("c2","c2",900,800)

    padUpper = ROOT.TPad('padUpper', 'padUpper', 0, 0.3, 1, 1.0)
    padUpper.SetBottomMargin(0.01)
    padUpper.SetTopMargin(0.12)
    #padUpper.SetLogy()
    padUpper.Draw()
    
    padLower = ROOT.TPad('padLower', 'padLower', 0, 0.0, 1, 0.3)
    padLower.SetBottomMargin(0.35)
    padLower.SetTopMargin(0.12)
    padLower.SetGridy()
    padLower.Draw()

    
    histoName = str(plotName)
    colorList = [ROOT.kRed, ROOT.kGreen, ROOT.kBlue, ROOT.kYellow, ROOT.kMagenta, ROOT.kCyan, ROOT.kOrange, ROOT.kSpring, ROOT.kTeal, ROOT.kAzure, ROOT.kViolet, ROOT.kPink]
    iColor = 0
    iLineStyle=0
    leg_dataVSmc = ROOT.TLegend(0.65, 0.75, 0.87, 0.87)
    leg_sigVSbkg = ROOT.TLegend(0.65, 0.75, 0.87, 0.87)
    
    histoStacked = ROOT.THStack("histoStacked","histoStacked")
    
    keyList_bkg = [key for key in config["background"]]
    inputHistoList_bkg = [ROOT.TH1D() for key in config["background"]]
    inputFileList_bkg = [ROOT.TFile() for key in config["background"]]
    inputHistoDic_bkg = dict(zip(keyList_bkg,inputHistoList_bkg))
    inputFileDic_bkg  = dict(zip(keyList_bkg,inputFileList_bkg))
    
    keyList_sig = [key for key in config["signal"]]
    inputHistoList_sig = [ROOT.TH1D() for key in config["signal"]]
    inputFileList_sig = [ROOT.TFile() for key in config["signal"]]
    inputHistoDic_sig = dict(zip(keyList_sig,inputHistoList_sig))
    inputFileDic_sig  = dict(zip(keyList_sig,inputFileList_sig))

    xaxis_label = str()
    
    #Stacking background histos
    for filename in config["background"]:
        inputFileDic_bkg [filename] = ROOT.TFile.Open(str(os.path.join(inputDirName,filename)))
        inputHistoDic_bkg[filename] = ROOT.TH1D(inputFileDic_bkg[filename].Get(histoName))
        integral = float(inputHistoDic_bkg[filename].Integral("width"))
        lumi_data = 0.000965888 #/fb
        xsec_mc = 0.8485e+12 #fb
        filter_eff = 0.002633
        nEvents_mc = 10000.
        nf_mc = lumi_data*xsec_mc*filter_eff/nEvents_mc
        #print("normalization factor: {}".format(nf_mc))
        inputHistoDic_bkg[filename].Scale(1./integral)
        #inputHistoDic_bkg[filename].Scale(nf_mc)
        inputHistoDic_bkg[filename].SetLineColor(ROOT.kBlack)
        inputHistoDic_bkg[filename].SetFillColor(colorList[iColor])
        xaxis_label = str(inputHistoDic_bkg[filename].GetXaxis().GetTitle())
        histoStacked.Add(inputHistoDic_bkg[filename])
        histoStacked.SetMinimum(10)
        iColor += 1
        bkgLabel = config["background"][filename]["label"]
        leg_dataVSmc.AddEntry(inputHistoDic_bkg[filename],bkgLabel)
        leg_sigVSbkg.AddEntry(inputHistoDic_bkg[filename],bkgLabel)

    c1.cd()
    hs = histoStacked.Clone()
    hs.Draw("hist")
    hs.SetTitle("")
    hs.GetYaxis().SetTitle("a.u.")
    hs.SetMinimum(0.)
    hs.SetMaximum(1.)
    hs.GetXaxis().SetTitle(xaxis_label)
    c1.Update()

    
    #Superimposing signal
    for filename in config["signal"]:
        inputFileDic_sig [filename] = ROOT.TFile.Open(str(os.path.join(inputDirName,filename)))
        inputHistoDic_sig[filename] = ROOT.TH1D(inputFileDic_sig[filename].Get(histoName))
        c1.cd()
        integral = float(inputHistoDic_sig[filename].Integral("width"))
        inputHistoDic_sig[filename].Scale(1./integral)
        inputHistoDic_sig[filename].SetLineColor(ROOT.kBlack)
        inputHistoDic_sig[filename].SetLineWidth(2)
        inputHistoDic_sig[filename].SetLineStyle(1+iLineStyle)
        inputHistoDic_sig[filename].SetMinimum(0.)
        inputHistoDic_sig[filename].SetMaximum(1.)
        #print("signal histo x axis: {}".format(inputHistoDic_sig[filename].GetXaxis().GetTitle()))
        sigLabel = config["signal"][filename]["label"]
        leg_sigVSbkg.AddEntry(inputHistoDic_sig[filename],sigLabel)
        inputHistoDic_sig[filename].Draw("histo same")
        iLineStyle+=1


    leg_sigVSbkg.Draw("same")
    c1.Update()

    #Drawing stacked histos
    padUpper.cd()
    histoStacked.SetTitle("")
    histoStacked.Draw("hist")
    histoStacked.GetYaxis().SetTitle("Events");
    histoStacked.GetYaxis().SetLabelSize(0.05);
    histoStacked.GetYaxis().SetTitleSize(0.06);
    histoStacked.GetYaxis().SetTitleOffset(0.8);
    histoStacked.GetXaxis().SetLabelSize(0);
    padUpper.Update()
    
    #Superimposing data
    inputDataFile = ROOT.TFile.Open(str(os.path.join(inputDirName,config["data"].keys()[0])))
    inputDataHisto = ROOT.TH1D(inputDataFile.Get(histoName))
    integral = float(inputDataHisto.Integral("width"))
    nf_data = 1.
    #inputDataHisto.Scale(nf_data)
    inputDataHisto.Scale(1./integral)
    padUpper.cd()
    inputDataHisto.SetLineColor(ROOT.kBlack)
    inputDataHisto.SetMarkerStyle(20)
    leg_dataVSmc.AddEntry(inputDataHisto,"DATA")
    inputDataHisto.Draw("same")
    leg_dataVSmc.Draw("same")
    
    inputMCFile = ROOT.TFile.Open(str(os.path.join(inputDirName,config["background"].keys()[0])))
    inputMCHisto = ROOT.TH1D(inputMCFile.Get(histoName))
    lumi_data = 0.000965888 #/fb
    xsec_mc = 0.8485e+12 #fb
    filter_eff = 0.002633
    nEvents_mc = 10000.
    nf_mc = lumi_data*xsec_mc*filter_eff/nEvents_mc
    integral = float(inputMCHisto.Integral("width"))
    #inputMCHisto.Scale(nf_mc)
    inputMCHisto.Scale(1./integral)

    hRatio = inputDataHisto.Clone()
    hRatio.Divide(inputMCHisto)
    padLower.cd()
    hRatio.SetTitle("")
    hRatio.GetYaxis().SetTitle("")
    hRatio.SetLineColor(ROOT.kBlack)
    hRatio.SetLineWidth(1)
    hRatio.SetMarkerStyle(20)
    hRatio.GetYaxis().SetRangeUser(0.5,1.5)
    hRatio.GetYaxis().SetTitle("DATA/MC")
    hRatio.GetXaxis().SetTitle(inputMCHisto.GetXaxis().GetTitle())
    hRatio.GetXaxis().SetTitleSize(0.12)
    hRatio.GetYaxis().SetTitleSize(0.12)
    hRatio.GetYaxis().SetLabelSize(0.12)
    hRatio.GetXaxis().SetLabelSize(0.12)
    hRatio.GetYaxis().SetTitleOffset(0.4)
    hRatio.GetYaxis().SetNdivisions(5)
    ROOT.gStyle.SetOptStat(0)
    hRatio.Draw()
    
    subprocess.call(["mkdir","-p",outDirName])
    c2.SaveAs(outDirName + "/" + histoName +"_dataVSmc.png")
    c2.SaveAs(outDirName + "/" + histoName +"_dataVSmc.root")
    c1.SaveAs(outDirName + "/" + histoName +"_sigVSbkg.png")
    c1.SaveAs(outDirName + "/" + histoName +"_sigVSbkg.root")
    del c2
