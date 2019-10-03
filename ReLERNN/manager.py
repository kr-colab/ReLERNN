'''
Author: Jeff Adrion

'''

from ReLERNN.imports import *
from ReLERNN.helpers import *

class Manager(object):
    '''

    The manager class is a framework for handling both VCFs and masks
    and can multi-process many of the functions orginally found in ReLERNN_SIMULATE

    '''


    def __init__(self,
        vcf = None,
        chromosomes = None,
        mask = None,
        winSizeMx = None,
        forceWinSize = None,
        vcfDir = None,
        projectDir = None,
        networkDir = None
        ):

        self.vcf = vcf
        self.chromosomes = chromosomes
        self.mask = mask
        self.winSizeMx = winSizeMx
        self.forceWinSize = forceWinSize
        self.vcfDir = vcfDir
        self.projectDir = projectDir
        self.networkDir = networkDir


    def splitVCF(self,nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.vcfDir, self.vcf, self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_splitVCF)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        return None


    def worker_splitVCF(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                vcfDir, vcf, chroms = params
                for i in mpID:
                    chrom = chroms[i].split(":")[0]
                    start = int(chroms[i].split(":")[1].split("-")[0])+1
                    end = int(chroms[i].split(":")[1].split("-")[1])+1
                    splitVCF=os.path.join(vcfDir, os.path.basename(vcf).replace(".vcf","_%s.vcf" %(chroms[i])))
                    print("Split chromosome: %s..." %(chrom))
                    with open(vcf, "r") as fIN, open(splitVCF, "w") as fOUT:
                        for line in fIN:
                            if line.startswith("#"):
                                fOUT.write(line)
                            if line.startswith("%s\t" %(chrom)):
                                pos = int(line.split()[1])
                                if start <= pos <= end:
                                    fOUT.write(line)
                    print("Converting %s to HDF5..." %(splitVCF))
                    h5FILE=splitVCF.replace(".vcf",".hdf5")
                    allel.vcf_to_hdf5(splitVCF,h5FILE,fields="*",overwrite=True)
                    os.system("rm %s" %(splitVCF))
            finally:
                task_q.task_done()


    def countSites(self, nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_countSites)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        wins = []
        for i in range(result_q.qsize()):
            item = result_q.get()
            wins.append(item)

        nSamps,maxS,maxLen = [],0,0
        sorted_wins = []
        winFILE=os.path.join(self.networkDir,"windowSizes.txt")
        with open(winFILE, "w") as fOUT:
            for chrom in self.chromosomes:
                for win in wins:
                    if win[0] == chrom:
                        maxS = max(maxS,win[4])
                        maxLen = max(maxLen,win[2])
                        nSamps.append(win[1])
                        sorted_wins.append(win)
                        fOUT.write("\t".join([str(x) for x in win])+"\n")
        if len(set(nSamps)) != 1:
            print("Error: chromosomes have different numbers of samples")
        return sorted_wins, nSamps[0], maxS, maxLen


    def worker_countSites(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                chromosomes = params
                for i in mpID:
                    h5FILE=os.path.join(self.vcfDir, os.path.basename(self.vcf).replace(".vcf","_%s.hdf5" %(chromosomes[i])))
                    print("""\nReading HDF5: "%s"...""" %(h5FILE))
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
                    if self.forceWinSize != 0:
                        ip = force_win_size(self.forceWinSize,pos)
                        result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
                    else:
                        step=1000
                        winSize=1000000
                        while winSize > 0:
                            ip = find_win_size(winSize,pos,step,self.winSizeMx)
                            if len(ip) != 5:
                                winSize-=step
                            else:
                                result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
                                winSize=0
            finally:
                task_q.task_done()


    def maskWins(self, maxLen=None, wins=None, nProc=1):
        '''
        split the vcf into seperate files by chromosome
        '''
        ## Read accessability mask
        print("Accessibility mask found: calculating the proportion of the genome that is masked...")
        mask={}
        with open(self.mask, "r") as fIN:
            for line in fIN:
                ar = line.split()
                try:
                    mask[ar[0]].append([int(pos) for pos in ar[1:]])
                except KeyError:
                    mask[ar[0]] = [[int(pos) for pos in ar[1:]]]

        ## Combine genomic windows
        genomic_wins = []
        for win in wins:
            win_chrom = win[0]
            win_len = win[2]
            win_ct = win[6]
            start = 0
            for i in range(win_ct):
                genomic_wins.append([win_chrom, start, win_len])
                start += win_len

        # partition for multiprocessing
        mpID = range(len(genomic_wins))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=genomic_wins, mask, maxLen

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_maskWins)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        masks = []
        for i in range(result_q.qsize()):
            item = result_q.get()
            masks.append(item)

        mask_fraction, win_masks = [], []
        for mask in masks:
            mask_fraction.append(mask[0])
            win_masks.append(mask)

        mean_mask_fraction = sum(mask_fraction)/float(len(mask_fraction))
        print("{}% of genome inaccessible".format(round(mean_mask_fraction * 100,1)))
        return mean_mask_fraction, win_masks


    def worker_maskWins(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                genomic_wins, mask, maxLen = params
                last_win = 0
                last_chrom = genomic_wins[0][0].split(":")[0]
                for i in mpID:
                    if genomic_wins[i][0].split(":")[0] != last_chrom:
                        last_win = 0
                        last_chrom = genomic_wins[i][0].split(":")[0]
                    M = maskStats(genomic_wins[i], last_win, mask, maxLen)
                    last_win = M[2]
                    result_q.put(M)
            finally:
                task_q.task_done()

