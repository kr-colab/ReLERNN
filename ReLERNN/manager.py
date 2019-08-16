'''
Author: Jeff Adrion

'''

from ReLERNN.imports import *

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
        vcfDir = None,
        projectDir = None,
        networkDir = None
        ):

        self.vcf = vcf
        self.chromosomes = chromosomes
        self.mask = mask
        self.winSizeMx = winSizeMx
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
        pids = self.create_procs_splitVCF(nProc, task_q, result_q, params)
        self.assign_task_splitVCF(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        return None


    def create_procs_splitVCF(self, nProcs, task_q, result_q, params):
        pids = []
        for _ in range(nProcs):
            p = mp.Process(target=self.worker_splitVCF, args=(task_q, result_q, params))
            p.daemon = True
            p.start()
            pids.append(p)
        return pids


    def assign_task_splitVCF(self, mpID, task_q, nProcs):
        c,i,nth_job=0,0,1
        while (i+1)*nProcs <= len(mpID):
            i+=1
        nP1=nProcs-(len(mpID)%nProcs)
        for j in range(nP1):
            task_q.put((mpID[c:c+i], nth_job))
            nth_job += 1
            c=c+i
        for j in range(nProcs-nP1):
            task_q.put((mpID[c:c+i+1], nth_job))
            nth_job += 1
            c=c+i+1


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
        pids = self.create_procs_countSites(nProc, task_q, result_q, params)
        self.assign_task_countSites(mpID, task_q, nProc)
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


    def snps_per_win(self, pos, window_size):
        bins = np.arange(1, pos.max()+window_size, window_size)
        y,x = np.histogram(pos,bins=bins)
        return y


    def find_win_size(self, winSize, pos, step, winSizeMx):
        snpsWin=self.snps_per_win(pos,winSize)
        mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
        if mx <= winSizeMx:
            return [winSize,mn,u,mx,len(snpsWin)]
        else:
            return [mn,u,mx]


    def create_procs_countSites(self, nProcs, task_q, result_q, params):
        pids = []
        for _ in range(nProcs):
            p = mp.Process(target=self.worker_countSites, args=(task_q, result_q, params))
            p.daemon = True
            p.start()
            pids.append(p)
        return pids


    def assign_task_countSites(self, mpID, task_q, nProcs):
        c,i,nth_job=0,0,1
        while (i+1)*nProcs <= len(mpID):
            i+=1
        nP1=nProcs-(len(mpID)%nProcs)
        for j in range(nP1):
            task_q.put((mpID[c:c+i], nth_job))
            nth_job += 1
            c=c+i
        for j in range(nProcs-nP1):
            task_q.put((mpID[c:c+i+1], nth_job))
            nth_job += 1
            c=c+i+1


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
                    step=1000
                    winSize=1000000
                    while winSize > 0:
                        ip = self.find_win_size(winSize,pos,step,self.winSizeMx)
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
        pids = self.create_procs_maskWins(nProc, task_q, result_q, params)
        self.assign_task_maskWins(mpID, task_q, nProc)
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

        print("{}% of genome inaccessible".format(round(self.mean(mask_fraction)*100,1)))
        return self.mean(mask_fraction), win_masks


    def maskStats(self, wins, last_win, mask, maxLen):
        """
        return a three-element list with the first element being the total proportion of the window that is masked,
        the second element being a list of masked positions that are relative to the windown start=0 and the window end = window length,
        and the third being the last window before breaking to expidite the next loop
        """
        chrom = wins[0].split(":")[0]
        a = wins[1]
        L = wins[2]
        b = a + L
        prop = [0.0,[],0]
        try:
            for i in range(last_win, len(mask[chrom])):
                x, y = mask[chrom][i][0], mask[chrom][i][1]
                if y < a:
                    continue
                if b < x:
                    return prop
                else:  # i.e. [a--b] and [x--y] overlap
                    if a >= x and b <= y:
                        return [1.0, [[0,maxLen]], i]
                    elif a >= x and b > y:
                        win_prop = (y-a)/float(b-a)
                        prop[0] += win_prop
                        prop[1].append([0,int(win_prop * maxLen)])
                        prop[2] = i
                    elif b <= y and a < x:
                        win_prop = (b-x)/float(b-a)
                        prop[0] += win_prop
                        prop[1].append([int((1-win_prop)*maxLen),maxLen])
                        prop[2] = i
                    else:
                        win_prop = (y-x)/float(b-a)
                        prop[0] += win_prop
                        prop[1].append([int(((x-a)/float(b-a))*maxLen), int(((y-a)/float(b-a))*maxLen)])
                        prop[2] = i
            return prop
        except KeyError:
            return prop


    def mean(self, L):
        return sum(L)/float(len(L))


    def create_procs_maskWins(self, nProcs, task_q, result_q, params):
        pids = []
        for _ in range(nProcs):
            p = mp.Process(target=self.worker_maskWins, args=(task_q, result_q, params))
            p.daemon = True
            p.start()
            pids.append(p)
        return pids


    def assign_task_maskWins(self, mpID, task_q, nProcs):
        c,i,nth_job=0,0,1
        while (i+1)*nProcs <= len(mpID):
            i+=1
        nP1=nProcs-(len(mpID)%nProcs)
        for j in range(nP1):
            task_q.put((mpID[c:c+i], nth_job))
            nth_job += 1
            c=c+i
        for j in range(nProcs-nP1):
            task_q.put((mpID[c:c+i+1], nth_job))
            nth_job += 1
            c=c+i+1


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
                    M = self.maskStats(genomic_wins[i], last_win, mask, maxLen)
                    last_win = M[2]
                    result_q.put(M)
            finally:
                task_q.task_done()

