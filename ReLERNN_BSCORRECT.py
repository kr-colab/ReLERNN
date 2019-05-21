"""
Performs a parametric bootstrap to assess any potential bias in recombination rate predictions.
Corrects for this bias and adds 95% confidence intevals to the predictions
"""


import os,sys
relernnBase = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),"scripts")
sys.path.insert(1, relernnBase)

from Imports import *
from Simulator import *
from Helpers import *
from SequenceBatchGenerator import *
from Networks import *


def relu(x):
    return max(0,x)


def get_index(L,N):
    idx,outN="",""
    dist=float("inf")
    for i in range(len(L)):
        D=abs(N-L[i])
        if D < dist:
            idx=i
            outN=L[i]
            dist=D
    return [idx,outN]


def get_corrected(rate,bs):
    idx=get_index(bs["Q2"],rate)
    CI95LO=bs["CI95LO"][idx[0]]
    CI95HI=bs["CI95HI"][idx[0]]
    cRATE=relu(rate+(bs["rho"][idx[0]]-idx[1]))
    ciHI=relu(cRATE+(CI95HI-idx[1]))
    ciLO=relu(cRATE+(CI95LO-idx[1]))
    return [cRATE,ciLO,ciHI]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--projectDir',dest='outDir',help='Directory for all project output. NOTE: the same projectDir must be used for all functions of ReLERNN')
    parser.add_argument('--gpuID',dest='gpuID',help='Identifier specifying which GPU to use', type=int, default=0)
    parser.add_argument('--nCPU',dest='nCPU',help='Number of CPUs to use', type=int, default=1)
    args = parser.parse_args()


    ## Set up the directory structure and output files
    DataDir = args.outDir
    trainDir = os.path.join(DataDir,"train")
    valiDir = os.path.join(DataDir,"vali")
    testDir = os.path.join(DataDir,"test")
    networkDir = os.path.join(DataDir,"networks")
    bs_resultFile = os.path.join(networkDir,"bootstrapResults.p")
    bs_plotFile = os.path.join(networkDir,"bootstrapPlot.pdf")
    modelWeights = [os.path.join(networkDir,"model.json"),os.path.join(networkDir,"weights.h5")]
    bs_resultFile = os.path.join(networkDir,"bootstrapResults.p")
    bsDir = os.path.join(DataDir,"PBS")


    ## Load simulation and batch pars
    simParsFILE=os.path.join(networkDir,"simPars.p")
    batchParsFILE=os.path.join(networkDir,"batchPars.p")
    with open(simParsFILE, "rb") as fIN:
        simPars=pickle.load(fIN)
    with open(batchParsFILE, "rb") as fIN:
        batchPars=pickle.load(fIN)
    pred_resultFiles = []
    for f in glob.glob(os.path.join(DataDir,"*.PREDICT.txt")):
        pred_resultFiles.append(f)
    if len(pred_resultFiles) < 1:
        print("Error: no .PREDICT.txt file found. You must run ReLERNN_PREDICT.py prior to running ReLERNN_BSCORRECT.py")
        sys.exit(1)
    elif len(pred_resultFiles) > 1:
        print("Error: multiple prediction files found.")
        sys.exit(1)
    pred_resultFile = pred_resultFiles[0]


    ## Strap it on!
    ParametricBootStrap(
            simPars,
            batchPars,
            trainDir,
            network=modelWeights,
            slices=100,
            repsPerSlice=1000,
            gpuID=args.gpuID,
            out=bs_resultFile,
            tempDir=bsDir,
            nCPU=args.nCPU)


    ## Plot results from bootstrap
    plotParametricBootstrap(bs_resultFile,bs_plotFile)


    ## Load bootstrap values
    with open(bs_resultFile, "rb") as fIN:
        bs=pickle.load(fIN)


    ## Loop, correct, and write output
    correctedfile=pred_resultFile.replace(".txt", ".BSCORRECTED.txt")
    with open(correctedfile, "w") as fout, open(pred_resultFile, "r") as fin:
        for line in fin:
            if not line.startswith("chrom"):
                ar=line.split()
                rate=float(ar[-1])
                C=get_corrected(rate,bs)
                ar[-1]=C[0]
                ar.extend([C[1],C[2]])
                fout.write("\t".join([str(x) for x in ar])+"\n")
            else:
                #fout.write(line)
                fout.write("%s\t%s\t%s\t%s\t%s\t%s\n" %("chrom","start","end","recombRate","CI95LO","CI95HI"))

    ## Remove the bootstrap tree files
    shutil.rmtree(bsDir)
    print("\n***ReLERNN_BSCORRECT.py FINISHED!***\n")


if __name__ == "__main__":
	main()
