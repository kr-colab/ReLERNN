"""
Reads a VCF file, estimates simulation parameters, and runs simulates via msprime.
NOTE: This assumes that the user has previously QC'd and filtered the VCF.
"""

import os,sys
relernnBase = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),"scripts")
sys.path.insert(1, relernnBase)

from Imports import *
from Simulator import *
from Helpers import *


def check_demHist(path):
    fTypeFlag = -9
    with open(path, "r") as fIN:
        for line in fIN:
            if line.startswith("mutation_per_site"):
                fTypeFlag = 1
                break
            if line.startswith("label") and "plot_type" in line:
                fTypeFlag = 2
                break
            if line.startswith("label") and not "plot_type" in line:
                fTypeFlag = 3
                break
    return fTypeFlag


def convert_demHist(path, nSamps, gen, fType):
    swp, PC, DE = [],[],[]
    # Convert stairwayplot to msp demographic_events
    if fType == 1:
        with open(path, "r") as fIN:
            flag=0
            lCt=0
            for line in fIN:
                if flag == 1:
                    if lCt % 2 == 0:
                        swp.append(line.split())
                    lCt+=1
                if line.startswith("mutation_per_site"):
                    flag=1
        N0 = int(float(swp[0][6]))
        for i in range(len(swp)):
            if i == 0:
                PC.append(msp.PopulationConfiguration(sample_size=nSamps, initial_size=N0))
            else:
                DE.append(msp.PopulationParametersChange(time=int(float(swp[i][5])/float(gen)), initial_size=int(float(swp[i][6])), population=0))
    ## Convert smc++ or MSMC results to msp demographic_events
    if fType == 2 or fType == 3:
        with open(path, "r") as fIN:
            fIN.readline()
            for line in fIN:
                ar=line.split(",")
                swp.append([int(float(ar[1])/gen),int(float(ar[2]))])
        N0 = swp[0][1]
        for i in range(len(swp)):
            if i == 0:
                PC.append(msp.PopulationConfiguration(sample_size=nSamps, initial_size=N0))
            else:
                DE.append(msp.PopulationParametersChange(time=swp[i][0], initial_size=swp[i][1], population=0))
    dd=msp.DemographyDebugger(population_configurations=PC,
            demographic_events=DE)
    print("Simulating under the following population size history:")
    dd.print_history()
    MspD = {"population_configurations" : PC,
        "migration_matrix" : None,
        "demographic_events" : DE}
    if MspD:
        return MspD
    else:
        print("Error in converting demographic history file.")
        sys.exit(1)


def split_VCF(D,vcf, basename, chroms):
    P=[]
    print("Spliting VCF...")
    for i in range(len(chroms)):
        print("Split chromosome: %s" %(chroms[i]))
        outFILE=os.path.join(D, basename.replace(".vcf","_%s.vcf" %(chroms[i])))
        P.append(outFILE)
        with open(vcf, "r") as fIN, open(outFILE, "w") as fOUT:
            for line in fIN:
                if line.startswith("#"):
                    fOUT.write(line)
                if line.startswith("%s\t" %(chroms[i])):
                    fOUT.write(line)
    return P


def snps_per_win(pos, window_size):
    bins = np.arange(1, pos.max()+window_size, window_size)
    y,x = np.histogram(pos,bins=bins)
    return y


def find_win_size(winSize,pos,step):
    snpsWin=snps_per_win(pos,winSize)
    mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
    if mx <= 1600:
        return [winSize,mn,u,mx,len(snpsWin)]
    else:
        return [mn,u,mx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--vcf',dest='vcf',help='Filtered and QC-checked VCF file Note: Every row must correspond to a biallelic SNP with no missing data)')
    parser.add_argument('-d','--projectDir',dest='outDir',help='Directory for all project output. NOTE: the same projectDir must be used for all functions of ReLERNN')
    parser.add_argument('-n','--demographicHistory',dest='dem',help='Output file from either stairwayplot, SMC++, or MSMC',default=0)
    parser.add_argument('-m','--assumedMu',dest='mu',help='Assumed per-base mutation rate',type=float,default=1e-8)
    parser.add_argument('-g','--assumedGenTime',dest='genTime',help='Assumed generation time (in years)',type=float)
    parser.add_argument('-r','--upperRhoThetaRatio',dest='upRTR',help='Upper bound for the assumed ratio between rho and theta',type=float,default=10)
    parser.add_argument('--nTrain',dest='nTrain',help='Number of training examples to simulate',type=int,default=100000)
    parser.add_argument('--nVali',dest='nVali',help='Number of validation examples to simulate',type=int,default=1000)
    parser.add_argument('--nTest',dest='nTest',help='Number of test examples to simulate',type=int,default=1000)
    parser.add_argument('-t','--nCPU',dest='nCPU',help='Number of CPUs to use',type=int,default=1)
    args = parser.parse_args()


    # Ensure all required arguments are provided
    if not args.vcf.endswith(".vcf"):
        print('Error: VCF file must end in extension ".vcf"')
        sys.exit(1)
    if not args.outDir:
        print("Error: Path to project directory required!")
        sys.exit(1)
    if args.dem != 0:
        demHist = check_demHist(args.dem)
        if demHist == -9:
            print("Error: demographicHistory file must be raw output from either stairwayplot, SMC++, or MSMC")
            sys.exit(1)
        if not args.genTime:
            print("Error: assumed generation time must be supplied when simulating under stairwayplot, SMC++, or MSMC")
            sys.exit(1)
    else:
        print("Warning: no demographic history file found. All training data will be simulated under demographic equilibrium.")
        demHist = 0


    ## Set up the directory structure to store the simulations data.
    nProc = args.nCPU
    DataDir = args.outDir
    trainDir = os.path.join(DataDir,"train")
    valiDir = os.path.join(DataDir,"vali")
    testDir = os.path.join(DataDir,"test")
    networkDir = os.path.join(DataDir,"networks")
    vcfDir = os.path.join(DataDir,"splitVCFs")
    swpDir = os.path.join(DataDir,"strwyplt")


    ## Make directories if they do not exist
    for p in [DataDir,trainDir,valiDir,testDir,networkDir,vcfDir]:
        if not os.path.exists(p):
            os.makedirs(p)


    ## Check for chromFILE, and create if it does not exist
    chromFILE=os.path.join(vcfDir, "chromFile.txt")
    chromosomes=[]
    if not os.path.exists(chromFILE):
        print("One-time read through VCF...")
        with open(chromFILE, "w") as fOUT, open(args.vcf, "r") as fIN:
            for line in fIN:
                if not line.startswith("#"):
                    ar=line.split()
                    if ar[0] not in chromosomes:
                        fOUT.write(ar[0]+"\n")
                        chromosomes.append(ar[0])
    else:
        with open(chromFILE, "r") as fIN:
            for line in fIN:
                ar=line.split()
                chromosomes.append(ar[0])


    ## Split VCF into separate chromosomes if not already split
    flag=0
    paths=[]
    bn=os.path.basename(args.vcf)
    for i in range(len(chromosomes)):
        tmp=os.path.join(vcfDir, bn.replace(".vcf","_%s.vcf" %(chromosomes[i])))
        if not os.path.exists(tmp):
            flag=1
            break
        else:
            paths.append(tmp)
    if flag==1:
        paths=split_VCF(vcfDir, args.vcf, bn, chromosomes)


    ## Convert to hdf5 file if this has not been done previously
    for i in range(len(paths)):
        inVCF=paths[i]
        h5FILE=inVCF.replace(".vcf",".hdf5")
        if not os.path.exists(h5FILE):
            print("Converting %s to HDF5..." %(inVCF))
            allel.vcf_to_hdf5(inVCF,h5FILE,fields="*")


    ## Read in the hdf5
    wins=[]
    winFILE=os.path.join(networkDir,"windowSizes.txt")
    if not os.path.exists(winFILE):
        with open(winFILE, "w") as fOUT:
            for i in range(len(paths)):
                inVCF=paths[i]
                h5FILE=inVCF.replace(".vcf",".hdf5")
                print("""\nImporting HDF5: "%s"...""" %(h5FILE))
                callset=h5py.File(h5FILE, mode="r")
                var=allel.VariantChunkedTable(callset["variants"],names=["CHROM","POS"], index="POS")
                chroms=var["CHROM"]
                pos=var["POS"]
                genos=allel.GenotypeChunkedArray(callset["calldata"]["GT"])


                #Is this a haploid or diploid VCF?
                GT=genos.to_haplotypes()
                GT=GT[:,1:2]
                GT=GT[0].tolist()
                if len(set(GT)) == 1 and GT[0] == -1:
                    nSamps=len(genos[0])
                else:
                    nSamps=len(genos[0])*2


                ## Identify ideal training parameters
                print("Finding best window size for chromosome: %s..." %(chroms[0]))
                step=1000
                winSize=100000
                while winSize > 0:
                    ip = find_win_size(winSize,pos,step)
                    if len(ip) != 5:
                        winSize-=step
                    else:
                        wins.append([chroms[0],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
                        fOUT.write(chroms[0]+"\t"+str(nSamps)+"\t"+"\t".join([str(x) for x in ip])+"\n")
                        winSize=0
    else:
        with open(winFILE, "r") as fIN:
            for line in fIN:
                ar=line.split()
                wins.append([ar[0],int(ar[1]),int(ar[2]),int(ar[3]),int(ar[4]),int(ar[5]),int(ar[6])])


    ## Tally chromosome estimates
    nSam=[]
    maxMean=0
    maxLen=0
    for i in range(len(wins)):
        maxMean=max([maxMean,wins[i][4]])
        maxLen=max([maxLen,wins[i][2]])
        nSam.append(wins[i][1])
    if len(set(nSam)) > 1:
        print(set(nSam))
        print("Error: Sample size differs among chromosomes!")
        sys.exit(1)


    ## Define parameters for msprime simulation
    nSamps=nSam[0]
    a=0
    for i in range(nSamps-1):
        a+=1/(i+1)
    theta=maxMean/a
    assumedMu = args.mu
    Ne=int(theta/(4.0 * assumedMu * maxLen))
    rhoHi=assumedMu*args.upRTR
    if demHist:
        MspD = convert_demHist(args.dem, nSamps, args.genTime, demHist)
        dg_params = {
                'priorLowsRho':0.0,
                'priorHighsRho':rhoHi,
                'priorLowsMu':assumedMu * 0.75,
                'priorHighsMu':assumedMu * 1.25,
                'ChromosomeLength':maxLen,
                'MspDemographics': MspD
                  }

    else:
        dg_params = {'N': nSamps,
            'Ne': Ne,
            'priorLowsRho':0.0,
            'priorHighsRho':rhoHi,
            'priorLowsMu':assumedMu * 0.75,
            'priorHighsMu':assumedMu * 1.25,
            'ChromosomeLength':maxLen
                  }

    # Assign pars for each simulation
    dg_train = Simulator(**dg_params)
    dg_vali = Simulator(**dg_params)
    dg_test = Simulator(**dg_params)


    ## Dump simulation pars for use with parametric bootstrap
    simParsFILE=os.path.join(networkDir,"simPars.p")
    with open(simParsFILE, "wb") as fOUT:
        dg_params["bn"]=bn.replace(".vcf","")
        pickle.dump(dg_params,fOUT)


    ## Simulate data
    print("\nTraining set:")
    dg_train.simulateAndProduceTrees(numReps=args.nTrain,direc=trainDir,simulator="msprime",nProc=nProc)
    print("Validation set:")
    dg_vali.simulateAndProduceTrees(numReps=args.nVali,direc=valiDir,simulator="msprime",nProc=nProc)
    print("Test set:")
    dg_test.simulateAndProduceTrees(numReps=args.nTest,direc=testDir,simulator="msprime",nProc=nProc)
    print("\nSIMULATIONS FINISHED!\n")


    ## Count number of segregating sites in simulation
    SS=[]
    maxSegSites = 0
    minSegSites = float("inf")
    for ds in [trainDir,valiDir,testDir]:
        DsInfoDir = pickle.load(open(os.path.join(ds,"info.p"),"rb"))
        SS.extend(DsInfoDir["segSites"])
        segSitesInDs = max(DsInfoDir["segSites"])
        segSitesInDsMin = min(DsInfoDir["segSites"])
        maxSegSites = max(maxSegSites,segSitesInDs)
        minSegSites = min(minSegSites,segSitesInDsMin)


    ## Compare counts of segregating sites between simulations and input VCF
    print("SANITY CHECK")
    print("====================")
    print("numSegSites\tMin\tMean\tMax")
    print("Simulated:\t%s\t%s\t%s" %(minSegSites, int(sum(SS)/float(len(SS))), maxSegSites))
    for i in range(len(wins)):
        print("InputVCF %s:\t%s\t%s\t%s" %(wins[i][0],wins[i][3],wins[i][4],wins[i][5]))
    print("\n\n***ReLERNN_SIMULATE.py FINISHED!***\n")


if __name__ == "__main__":
	main()
