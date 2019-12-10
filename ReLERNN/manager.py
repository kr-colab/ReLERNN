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
        pool = None,
        chromosomes = None,
        mask = None,
        winSizeMx = None,
        forceWinSize = None,
        vcfDir = None,
        poolDir = None,
        projectDir = None,
        networkDir = None
        ):

        self.vcf = vcf
        self.pool = pool
        self.chromosomes = chromosomes
        self.mask = mask
        self.winSizeMx = winSizeMx
        self.forceWinSize = forceWinSize
        self.vcfDir = vcfDir
        self.poolDir = poolDir
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


    def splitPOOL(self,nProc=1):
        '''
        split the pool file into seperate files by chromosome
        '''
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.poolDir, self.pool, self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_splitPOOL)
        assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        return None


    def worker_splitPOOL(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                poolDir, pool, chroms = params
                for i in mpID:
                    chrom = chroms[i].split(":")[0]
                    start = int(chroms[i].split(":")[1].split("-")[0])+1
                    end = int(chroms[i].split(":")[1].split("-")[1])+1
                    splitPOOL=os.path.join(poolDir, os.path.basename(pool).replace(".pool","_%s.pool" %(chroms[i])))
                    print("Split chromosome: %s..." %(chrom))
                    with open(pool, "r") as fIN, open(splitPOOL, "w") as fOUT:
                        for line in fIN:
                            if line.startswith("%s\t" %(chrom)):
                                pos = int(line.split()[1])
                                if start <= pos <= end:
                                    fOUT.write(line)
            finally:
                task_q.task_done()


    def countSites(self, nProc=1):
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
                    GT=genos.to_haplotypes()
                    GTB=GT[:,1:2]
                    GTB=GTB[0].tolist()

                    if len(set(GTB)) == 1 and GTB[0] == -1:
                        nSamps=len(genos[0])
                        GT=GT[:,::2] #Select only the first of the genotypes
                    else:
                        GT=np.array(GT)
                        nSamps=len(genos[0])*2

                    ## if there is any missing data write a missing data boolean mask to hdf5
                    md_mask = GT < 0
                    if md_mask.any():
                        md_maskFile=os.path.join(self.vcfDir, os.path.basename(self.vcf).replace(".vcf","_%s_md_mask.hdf5" %(chromosomes[i])))
                        with h5py.File(md_maskFile, "w") as hf:
                            hf.create_dataset("mask", data=md_mask)

                    ## Find best window size
                    if self.forceWinSize != 0:
                        ip = force_win_size(self.forceWinSize,pos)
                        result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
                    else:
                        last, next_size = 0, round(int(chromosomes[i].split(":")[-1].split("-")[-1]),-3)
                        lo, hi = [last], [next_size]
                        D = next_size
                        while D > 5:
                            ip = find_win_size(next_size,pos,self.winSizeMx)
                            D = min(hi)-max(lo)
                            if len(ip) != 5:
                                last = next_size
                                if ip[0] < 0:
                                    hi.append(last)
                                if ip[0] > 0:
                                    lo.append(last)
                                next_size = max(lo) + int((min(hi) - max(lo))/2.0)
                            else:
                                break
                        next_size = round(next_size,-2)
                        ip = force_win_size(next_size,pos)
                        result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
            finally:
                task_q.task_done()


    def countSitesPOOL(self, samD=0, nProc=1):
        # partition for multiprocessing
        mpID = range(len(self.chromosomes))
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=self.chromosomes

        # do the work
        pids = create_procs(nProc, task_q, result_q, params, self.worker_countSitesPOOL)
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
                        maxS = max(maxS,win[3])
                        maxLen = max(maxLen,win[1])
                        win.insert(1,samD)
                        nSamps.append(samD)
                        sorted_wins.append(win)
                        fOUT.write("\t".join([str(x) for x in win])+"\n")
        if len(set(nSamps)) != 1:
            print("Error: chromosomes have different numbers of samples")
        return sorted_wins, nSamps[0], maxS, maxLen


    def worker_countSitesPOOL(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                chromosomes = params
                for i in mpID:
                    pos = []
                    poolFILE=os.path.join(self.poolDir, os.path.basename(self.pool).replace(".pool","_%s.pool" %(chromosomes[i])))
                    print("poolFILE:",poolFILE)
                    with open(poolFILE, "r") as fIN:
                        for line in fIN:
                            pos.append(int(line.split()[1]))
                    pos=np.array(pos)

                    ## Find best window size
                    if self.forceWinSize != 0:
                        ip = force_win_size(self.forceWinSize,pos)
                        result_q.put([chromosomes[i],ip[0],ip[1],ip[2],ip[3],ip[4]])
                    else:
                        last, next_size = 0, round(int(chromosomes[i].split(":")[-1].split("-")[-1]),-3)
                        lo, hi = [last], [next_size]
                        D = next_size
                        while D > 5:
                            ip = find_win_size(next_size,pos,self.winSizeMx)
                            D = min(hi)-max(lo)
                            if len(ip) != 5:
                                last = next_size
                                if ip[0] < 0:
                                    hi.append(last)
                                if ip[0] > 0:
                                    lo.append(last)
                                next_size = max(lo) + int((min(hi) - max(lo))/2.0)
                            else:
                                break
                        next_size = round(next_size,-2)
                        ip = force_win_size(next_size,pos)
                        result_q.put([chromosomes[i],nSamps,ip[0],ip[1],ip[2],ip[3],ip[4]])
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

