'''
Authors: Jared Galloway, Jeff Adrion
'''

from ReLERNN.imports import *

class SequenceBatchGenerator(tf.keras.utils.Sequence):

    '''
    This class, SequenceBatchGenerator, extends tf.keras.utils.Sequence.
    So as to multithread the batch preparation in tandum with network training
    for maximum effeciency on the hardware provided.

    It generated batches of genotype matrices from a given .trees directory
    (which is generated most effeciently from the Simulator class)
    which have been prepped according to the given parameters.

    It also offers a range of data prepping heuristics as well as normalizing
    the targets.

    def __getitem__(self, idx):

    def __data_generation(self, batchTreeIndices):

    '''

    #Initialize the member variables which largely determine the data prepping heuristics
    #in addition to the .trees directory containing the data from which to generate the batches
    def __init__(self,
            treesDirectory,
            targetNormalization = 'zscore',
            batchSize=64,
            maxLen=None,
            frameWidth=0,
            center=False,
            shuffleInds=False,
            sortInds=False,
            ancVal = -1,
            padVal = -1,
            derVal = 1,
            realLinePos = True,
            posPadVal = 0,
            shuffleExamples = True,
            splitFLAG = False,
            seqD = None,
            maf = None,
            hotspots = False
            ):

        self.treesDirectory = treesDirectory
        self.targetNormalization = targetNormalization
        infoFilename = os.path.join(self.treesDirectory,"info.p")
        self.infoDir = pickle.load(open(infoFilename,"rb"))
        self.batch_size = batchSize
        self.maxLen = maxLen
        self.frameWidth = frameWidth
        self.center = center
        self.shuffleInds = shuffleInds
        self.sortInds=sortInds
        self.ancVal = ancVal
        self.padVal = padVal
        self.derVal = derVal
        self.realLinePos = realLinePos
        self.posPadVal = posPadVal
        self.indices = np.arange(self.infoDir["numReps"])
        self.shuffleExamples = shuffleExamples
        self.splitFLAG = splitFLAG
        self.seqD = seqD
        self.maf = maf
        self.hotspots = hotspots

        if(targetNormalization != None):
            if self.hotspots:
                self.normalizedTargets = self.normalizeTargetsBinaryClass()
            else:
                self.normalizedTargets = self.normalizeTargets()

        if(shuffleExamples):
            np.random.shuffle(self.indices)

    def sort_min_diff(self,amat):
        '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
        this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
        assumes your input matrix is a numpy array'''

        mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
        v = mb.kneighbors(amat)
        smallest = np.argmin(v[0].sum(axis=1))
        return amat[v[1][smallest]]

    def pad_HapsPos(self,haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
        '''
        pads the haplotype and positions tensors
        to be uniform with the largest tensor
        '''

        haps = haplotypes
        pos = positions

        #Normalize the shape of all haplotype vectors with padding
        for i in range(len(haps)):
            numSNPs = haps[i].shape[0]
            paddingLen = maxSNPs - numSNPs
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                haps[i] = np.pad(haps[i],((prior,post),(0,0)),"constant",constant_values=2.0)
                pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

            else:
                if(paddingLen < 0):
                    haps[i] = np.pad(haps[i],((0,0),(0,0)),"constant",constant_values=2.0)[:paddingLen]
                    pos[i] = np.pad(pos[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                else:
                    haps[i] = np.pad(haps[i],((0,paddingLen),(0,0)),"constant",constant_values=2.0)
                    pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

        haps = np.array(haps,dtype='float32')
        pos = np.array(pos,dtype='float32')

        if(frameWidth):
            fw = frameWidth
            haps = np.pad(haps,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=2.0)
            pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)

        return haps,pos

    def padAlleleFqs(self,haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
        '''
        convert haps to allele frequencies, normalize, and
        pad the haplotype and positions tensors
        to be uniform with the largest tensor
        '''

        haps = haplotypes
        positions = positions
        fqs, pos = [], []

        # Resample to sequencing depth and convert to allele frequencies
        for i in range(len(haps)):
            tmp_freqs = []
            tmp_pos = []
            fqs_list = haps[i].tolist()
            for j in range(len(fqs_list)):

                if self.seqD != -9:
                    ## Resample
                    z = resample(fqs_list[j], n_samples=self.seqD, replace=True)
                    raw_freq = round(np.count_nonzero(z)/float(len(z)),3)
                    if self.maf <= raw_freq < 1.0:
                        tmp_freqs.append(raw_freq)
                        tmp_pos.append(positions[i][j])
                else:
                    ## Don't resample
                    raw_freq = round(np.count_nonzero(fqs_list[j])/float(len(fqs_list[j])),3)
                    tmp_freqs.append(raw_freq)
                    tmp_pos.append(positions[i][j])

            fqs.append(np.array(tmp_freqs))
            pos.append(np.array(tmp_pos))

        # Normalize
        fqs = self.normalizeAlleleFqs(fqs)

        # Pad
        for i in range(len(fqs)):
            numSNPs = fqs[i].shape[0]
            paddingLen = maxSNPs - numSNPs
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                fqs[i] = np.pad(fqs[i],(prior,post),"constant",constant_values=-1.0)
                pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

            else:
                if(paddingLen < 0):
                    fqs[i] = np.pad(fqs[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                    pos[i] = np.pad(pos[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                else:
                    fqs[i] = np.pad(fqs[i],(0,paddingLen),"constant",constant_values=-1.0)
                    pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

        fqs = np.array(fqs,dtype='float32')
        pos = np.array(pos,dtype='float32')

        if(frameWidth):
            fw = frameWidth
            fqs = np.pad(fqs,((0,0),(fw,fw)),"constant",constant_values=-1.0)
            pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)

        return fqs,pos

    def normalizeTargets(self):

        '''
        We want to normalize all targets.
        '''

        norm = self.targetNormalization
        nTargets = copy.deepcopy(self.infoDir['rho'])

        if(norm == 'zscore'):
            tar_mean = np.mean(nTargets,axis=0)
            tar_sd = np.std(nTargets,axis=0)
            nTargets -= tar_mean
            nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)

        elif(norm == 'divstd'):
            tar_sd = np.std(nTargets,axis=0)
            nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)

        return nTargets

    def normalizeTargetsBinaryClass(self):

        '''
        We want to normalize all targets.
        '''

        norm = self.targetNormalization
        nTargets = copy.deepcopy(self.infoDir['hotWin'])

        nTargets[nTargets<5] = 0
        nTargets[nTargets>=5] = 1

        return nTargets.astype(np.uint8)

    def normalizeAlleleFqs(self, fqs):

        '''
        normalize the allele frequencies for the batch
        '''

        norm = self.targetNormalization

        if(norm == 'zscore'):
            allVals = np.concatenate([a.flatten() for a in fqs])
            fqs_mean = np.mean(allVals)
            fqs_sd = np.std(allVals)
            for i in range(len(fqs)):
                fqs[i] = np.subtract(fqs[i],fqs_mean)
                fqs[i] = np.divide(fqs[i],fqs_sd,out=np.zeros_like(fqs[i]),where=fqs_sd!=0)

        elif(norm == 'divstd'):
            allVals = np.concatenate([a.flatten() for a in fqs])
            fqs_sd = np.std(allVals)
            for i in range(len(fqs)):
                fqs[i] = np.divide(fqs[i],fqs_sd,out=np.zeros_like(fqs[i]),where=fqs_sd!=0)

        return fqs

    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)

    def __len__(self):

        return int(np.floor(self.infoDir["numReps"]/self.batch_size))

    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X,y

    def shuffleIndividuals(self,x):
        t = np.arange(x.shape[1])
        np.random.shuffle(t)
        return x[:,t]

    def __data_generation(self, batchTreeIndices):

        haps = []
        pos = []

        for treeIndex in batchTreeIndices:
            Hfilepath = os.path.join(self.treesDirectory,str(treeIndex) + "_haps.npy")
            Pfilepath = os.path.join(self.treesDirectory,str(treeIndex) + "_pos.npy")
            H = np.load(Hfilepath)
            P = np.load(Pfilepath)
            haps.append(H)
            pos.append(P)

        respectiveNormalizedTargets = [[t] for t in self.normalizedTargets[batchTreeIndices]]
        targets = np.array(respectiveNormalizedTargets)

        if(self.realLinePos):
            for p in range(len(pos)):
                pos[p] = pos[p] / self.infoDir["ChromosomeLength"]

        if(self.sortInds):
            for i in range(len(haps)):
                haps[i] = np.transpose(self.sort_min_diff(np.transpose(haps[i])))

        if(self.shuffleInds):
            for i in range(len(haps)):
                haps[i] = self.shuffleIndividuals(haps[i])

        if self.seqD:
            # simulate pool-sequencing
            if(self.maxLen != None):
                # convert the haps to allele frequecies and then pad
                haps,pos = self.padAlleleFqs(haps,pos,
                    maxSNPs=self.maxLen,
                    frameWidth=self.frameWidth,
                    center=self.center)

                haps=np.where(haps == -1.0, self.posPadVal,haps)
                pos=np.where(pos == -1.0, self.posPadVal,pos)
                z = np.stack((haps,pos), axis=-1)

                return z, targets
        else:
            if(self.maxLen != None):
                # pad
                haps,pos = self.pad_HapsPos(haps,pos,
                    maxSNPs=self.maxLen,
                    frameWidth=self.frameWidth,
                    center=self.center)

                pos=np.where(pos == -1.0, self.posPadVal,pos)
                haps=np.where(haps < 1.0, self.ancVal, haps)
                haps=np.where(haps > 1.0, self.padVal, haps)
                haps=np.where(haps == 1.0, self.derVal, haps)

                return [haps,pos], targets


class VCFBatchGenerator(tf.keras.utils.Sequence):
    """Basically same as SequenceBatchGenerator Class except for VCF files"""
    def __init__(self,
            INFO,
            CHROM,
            WIN,
            IDs,
            GT,
            POS,
            batchSize=64,
            maxLen=None,
            frameWidth=0,
            center=False,
            sortInds=False,
            ancVal = -1,
            padVal = -1,
            derVal = 1,
            realLinePos = True,
            posPadVal = 0,
            phase=None
            ):

        self.INFO=INFO
        self.CHROM=CHROM
        self.WIN=WIN
        self.IDs=IDs
        self.GT=GT
        self.POS=POS
        self.batch_size = batchSize
        self.maxLen = maxLen
        self.frameWidth = frameWidth
        self.center = center
        self.sortInds=sortInds
        self.ancVal = ancVal
        self.padVal = padVal
        self.derVal = derVal
        self.realLinePos = realLinePos
        self.posPadVal = posPadVal
        self.phase=phase


    def pad_HapsPosVCF(self,haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
        '''
        pads the haplotype and positions tensors
        to be uniform with the largest tensor
        '''

        haps = haplotypes
        pos = positions

        nSNPs=[]

        #Normalize the shape of all haplotype vectors with padding
        for i in range(len(haps)):
            numSNPs = haps[i].shape[0]
            nSNPs.append(numSNPs)
            paddingLen = maxSNPs - numSNPs
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                haps[i] = np.pad(haps[i],((prior,post),(0,0)),"constant",constant_values=2.0)
                pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

            else:
                haps[i] = np.pad(haps[i],((0,paddingLen),(0,0)),"constant",constant_values=2.0)
                pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

        haps = np.array(haps,dtype='float32')
        pos = np.array(pos,dtype='float32')

        if(frameWidth):
            fw = frameWidth
            haps = np.pad(haps,((0,0),(fw,fw),(fw,fw)),"constant",constant_values=2.0)
            pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)
        return haps,pos,nSNPs

    def __getitem__(self, idx):
        genos=self.GT
        GT=self.GT.to_haplotypes()
        diploid_check=[]
        for n in range(1,len(genos[0]),2):
            GTB=GT[:,n:n+1]
            if np.unique(GTB).shape[0] == 1 and np.unique(GTB)[0] == -1:
                diploid_check.append(0)
            else:
                diploid_check.append(1)
                break
        if 1 in diploid_check:
            GT=np.array(GT)
        else:
            GT=GT[:,::2] #Select only the first of the genotypes
        GT = np.where(GT == -1, 2, GT) # Code missing data as 2, these will ultimately end up being transformed to the pad value

        if not self.phase:
            np.random.shuffle(np.transpose(GT))

        haps,pos=[],[]
        for i in range(len(self.IDs)):
            haps.append(GT[self.IDs[i][0]:self.IDs[i][1]])
            pos.append(self.POS[self.IDs[i][0]:self.IDs[i][1]])

        if(self.realLinePos):
            for i in range(len(pos)):
                pos[i] = (pos[i]-(self.WIN*i)) / self.WIN

        if(self.sortInds):
            for i in range(len(haps)):
                haps[i] = np.transpose(sort_min_diff(np.transpose(haps[i])))

        if(self.maxLen != None):
            ##then we're probably padding
            haps,pos,nSNPs = self.pad_HapsPosVCF(haps,pos,
                maxSNPs=self.maxLen,
                frameWidth=self.frameWidth,
                center=self.center)

            pos=np.where(pos == -1.0, self.posPadVal,pos)
            haps=np.where(haps < 1.0, self.ancVal, haps)
            haps=np.where(haps > 1.0, self.padVal, haps)
            haps=np.where(haps == 1.0, self.derVal, haps)

            return [haps,pos], self.CHROM, self.WIN, self.INFO, nSNPs


class POOLBatchGenerator(tf.keras.utils.Sequence):
    """Basically same as SequenceBatchGenerator Class except for POOL files"""
    def __init__(self,
            INFO,
            CHROM,
            WIN,
            IDs,
            GT,
            POS,
            batchSize=64,
            maxLen=None,
            frameWidth=0,
            center=False,
            sortInds=False,
            ancVal = -1,
            padVal = -1,
            derVal = 1,
            realLinePos = True,
            posPadVal = 0,
            normType = 'zscore',
            ):

        self.INFO=INFO
        self.normType = normType
        self.CHROM=CHROM
        self.WIN=WIN
        self.IDs=IDs
        self.GT=GT
        self.POS=POS
        self.batch_size = batchSize
        self.maxLen = maxLen
        self.frameWidth = frameWidth
        self.center = center
        self.sortInds=sortInds
        self.ancVal = ancVal
        self.padVal = padVal
        self.derVal = derVal
        self.realLinePos = realLinePos
        self.posPadVal = posPadVal


    def padFqs(self,haplotypes,positions,maxSNPs=None,frameWidth=0,center=False):
        '''
        normalize, and pad the haplotype and positions tensors
        to be uniform with the largest tensor
        '''

        fqs = haplotypes
        pos = positions

        # Normalize
        fqs = self.normalizeAlleleFqs(fqs)

        nSNPs=[]
        # Pad
        for i in range(len(fqs)):
            numSNPs = fqs[i].shape[0]
            nSNPs.append(numSNPs)
            paddingLen = maxSNPs - numSNPs
            if(center):
                prior = paddingLen // 2
                post = paddingLen - prior
                fqs[i] = np.pad(fqs[i],(prior,post),"constant",constant_values=-1.0)
                pos[i] = np.pad(pos[i],(prior,post),"constant",constant_values=-1.0)

            else:
                if(paddingLen < 0):
                    fqs[i] = np.pad(fqs[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                    pos[i] = np.pad(pos[i],(0,0),"constant",constant_values=-1.0)[:paddingLen]
                else:
                    fqs[i] = np.pad(fqs[i],(0,paddingLen),"constant",constant_values=-1.0)
                    pos[i] = np.pad(pos[i],(0,paddingLen),"constant",constant_values=-1.0)

        fqs = np.array(fqs,dtype='float32')
        pos = np.array(pos,dtype='float32')

        if(frameWidth):
            fw = frameWidth
            fqs = np.pad(fqs,((0,0),(fw,fw)),"constant",constant_values=-1.0)
            pos = np.pad(pos,((0,0),(fw,fw)),"constant",constant_values=-1.0)

        return fqs,pos,nSNPs


    def normalizeAlleleFqs(self, fqs):

        '''
        normalize the allele frequencies for the batch
        '''

        norm = self.normType

        if(norm == 'zscore'):
            allVals = np.concatenate([a.flatten() for a in fqs])
            fqs_mean = np.mean(allVals)
            fqs_sd = np.std(allVals)
            for i in range(len(fqs)):
                fqs[i] = np.subtract(fqs[i],fqs_mean)
                fqs[i] = np.divide(fqs[i],fqs_sd,out=np.zeros_like(fqs[i]),where=fqs_sd!=0)

        elif(norm == 'divstd'):
            allVals = np.concatenate([a.flatten() for a in fqs])
            fqs_sd = np.std(allVals)
            for i in range(len(fqs)):
                fqs[i] = np.divide(fqs[i],fqs_sd,out=np.zeros_like(fqs[i]),where=fqs_sd!=0)

        return fqs


    def __getitem__(self, idx):
        GT=self.GT

        haps,pos=[],[]
        for i in range(len(self.IDs)):
            haps.append(GT[self.IDs[i][0]:self.IDs[i][1]])
            pos.append(self.POS[self.IDs[i][0]:self.IDs[i][1]])

        if(self.realLinePos):
            for i in range(len(pos)):
                pos[i] = (pos[i]-(self.WIN*i)) / self.WIN

        if(self.sortInds):
            for i in range(len(haps)):
                haps[i] = np.transpose(sort_min_diff(np.transpose(haps[i])))

        # pad the allele freqs and positions
        if(self.maxLen != None):
            haps,pos,nSNPs = self.padFqs(haps,pos,
                maxSNPs=self.maxLen,
                frameWidth=self.frameWidth,
                center=self.center)

            haps=np.where(haps == -1.0, self.posPadVal,haps)
            pos=np.where(pos == -1.0, self.posPadVal,pos)
            np.set_printoptions(threshold=sys.maxsize)
            z = np.stack((haps,pos), axis=-1)

            return z, self.CHROM, self.WIN, self.INFO, nSNPs


