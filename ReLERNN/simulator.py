'''
Author: Jared Galloway, Jeff Adrion

'''

from ReLERNN.imports import *

class Simulator(object):
    '''

    The simulator class is a framework for running N simulations
    using Either msprime (coalescent) or SLiM (forward-moving)
    in parallel using python's multithreading package.

    With Specified parameters, the class Simulator() populates
    a directory with training, validation, and testing datasets.
    It stores the the treeSequences resulting from each simulation
    in a subdirectory respectfully labeled 'i.trees' where i is the
    i^th simulation.

    Included with each dataset this class produces an info.p
    in the subdirectory. This uses pickle to store a dictionary
    containing all the information for each simulation including the random
    target parameter which will be extracted for training.

    '''

    def __init__(self,
        N = 2,
	Ne = 1e2,
        priorLowsRho = 0.0,
        priorLowsMu = 0.0,
        priorHighsRho = 1e-7,
        priorHighsMu = 1e-8,
        ChromosomeLength = 1e5,
        MspDemographics = None,
        winMasks = None,
        maskThresh = 1.0,
        phased = None,
        phaseError = None
        ):

        self.N = N
        self.Ne = Ne
        self.priorLowsRho = priorLowsRho
        self.priorHighsRho = priorHighsRho
        self.priorLowsMu = priorLowsMu
        self.priorHighsMu = priorHighsMu
        self.ChromosomeLength = ChromosomeLength
        self.MspDemographics = MspDemographics
        self.rho = None
        self.mu = None
        self.segSites = None
        self.winMasks = winMasks
        self.maskThresh = maskThresh
        self.phased = None
        self.phaseError = phaseError


    def runOneMsprimeSim(self,simNum,direc):
        '''
        run one msprime simulation and put the corresponding treeSequence in treesOutputFilePath

        (str,float,float)->None
        '''

        MR = self.mu[simNum]
        RR = self.rho[simNum]

        if self.MspDemographics:
            DE = self.MspDemographics["demographic_events"]
            PC = self.MspDemographics["population_configurations"]
            MM = self.MspDemographics["migration_matrix"]
            ts = msp.simulate(
                length=self.ChromosomeLength,
                mutation_rate=MR,
                recombination_rate=RR,
                population_configurations = PC,
                migration_matrix = MM,
                demographic_events = DE
            )
        else:
            ts = msp.simulate(
                sample_size = self.N,
                Ne = self.Ne,
                length=self.ChromosomeLength,
                mutation_rate=MR,
                recombination_rate=RR
            )

        # Convert tree sequence to genotype matrix, and position matrix
        H = ts.genotype_matrix()
        P = np.array([s.position for s in ts.sites()],dtype='float32')

        # "Unphase" genotypes
        if not self.phased:
            np.random.shuffle(np.transpose(H))

        # Simulate phasing error
        if self.phaseError:
            H = self.phaseErrorer(H,self.phaseError)

        # Sample from the genome-wide distribution of masks and mask both positions and genotypes
        if self.winMasks:
            while True:
                rand_mask = self.winMasks[random.randint(0,len(self.winMasks)-1)]
                if rand_mask[0] < self.maskThresh:
                    break
            if rand_mask[0] > 0.0:
                H,P = self.maskGenotypes(H, P, rand_mask)

        # Dump
        Hname = str(simNum) + "_haps.npy"
        Hpath = os.path.join(direc,Hname)
        Pname = str(simNum) + "_pos.npy"
        Ppath = os.path.join(direc,Pname)
        np.save(Hpath,H)
        np.save(Ppath,P)

        # Return number of sites
        return H.shape[0]


    def maskGenotypes(self, H, P, rand_mask):
        """
        Return the genotype and position matrices where masked sites have been removed
        """
        mask_wins = np.array(rand_mask[1])
        mask_wins = np.reshape(mask_wins, 2 * len(mask_wins))
        mask = np.digitize(P, mask_wins) % 2 == 0
        return H[mask], P[mask]


    def phaseErrorer(self, H, rate):
        """
        Returns the genotype matrix where some fraction of sites have shuffled samples
        """
        H_shuf = copy.deepcopy(H)
        np.random.shuffle(np.transpose(H_shuf))
        H_mask = np.random.choice([True,False], H.shape[0], p = [1-rate,rate])
        H_mask = np.repeat(H_mask, H.shape[1])
        H_mask = H_mask.reshape(H.shape)
        return np.where(H_mask,H,H_shuf)


    def simulateAndProduceTrees(self,direc,numReps,simulator,nProc=1):
        '''
        determine which simulator to use then populate

        (str,str) -> None
        '''
        self.rho=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.priorLowsRho,self.priorHighsRho)
            self.rho[i] = randomTargetParameter

        self.mu=np.empty(numReps)
        for i in range(numReps):
            randomTargetParameter = np.random.uniform(self.priorLowsMu,self.priorHighsMu)
            self.mu[i] = randomTargetParameter

        try:
            assert((simulator=='msprime') | (simulator=='SLiM'))
        except:
            print("Sorry, only 'msprime' & 'SLiM' are supported simulators")
            exit()

        #Pretty straitforward, create the directory passed if it doesn't exits
        if not os.path.exists(direc):
            print("directory '",direc,"' does not exist, creating it")
            os.makedirs(direc)

        # partition data for multiprocessing
        mpID = range(numReps)
        task_q = mp.JoinableQueue()
        result_q = mp.Queue()
        params=[simulator, direc]

        # do the work
        print("Simulate...")
        pids = self.create_procs(nProc, task_q, result_q, params)
        self.assign_task(mpID, task_q, nProc)
        try:
            task_q.join()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            sys.exit(0)

        self.segSites=np.empty(numReps,dtype="int64")
        for i in range(result_q.qsize()):
            item = result_q.get()
            self.segSites[item[0]]=item[1]

        self.__dict__["numReps"] = numReps
        infofile = open(os.path.join(direc,"info.p"),"wb")
        pickle.dump(self.__dict__,infofile)
        infofile.close()

        for p in pids:
            p.terminate()
        return None


    def assign_task(self, mpID, task_q, nProcs):
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


    def create_procs(self, nProcs, task_q, result_q, params):
        pids = []
        for _ in range(nProcs):
            p = mp.Process(target=self.worker, args=(task_q, result_q, params))
            p.daemon = True
            p.start()
            pids.append(p)
        return pids


    def worker(self, task_q, result_q, params):
        while True:
            try:
                mpID, nth_job = task_q.get()
                #unpack parameters
                simulator, direc = params
                for i in mpID:
                        result_q.put([i,self.runOneMsprimeSim(i,direc)])
            finally:
                task_q.task_done()

