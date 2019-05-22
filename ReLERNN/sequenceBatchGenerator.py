'''
Authors: Jared Galloway, Jeff Adrion
'''

from ReLERNN.imports import *
from ReLERNN.helpers import *

class SequenceBatchGenerator(keras.utils.Sequence):

    '''
    This class, SequenceBatchGenerator, extends keras.utils.Sequence.
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
            ):

        self.treesDirectory = treesDirectory
        self.targetNormalization = targetNormalization
        infoFilename = os.path.join(self.treesDirectory,"info.p")
        self.infoDir = pickle.load(open(infoFilename,"rb"))
        if(targetNormalization != None):
            self.normalizedTargets = self.normalizeTargets()
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

        if(shuffleExamples):
            np.random.shuffle(self.indices)

    def shuffleIndividuals(self,x):
        t = np.arange(x.shape[1])
        np.random.shuffle(t)
        return x[:,t]

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

    def on_epoch_end(self):

        if(self.shuffleExamples):
            np.random.shuffle(self.indices)

    def __len__(self):

        return int(np.floor(self.infoDir["numReps"]/self.batch_size))

    def __getitem__(self, idx):

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X,y

    def __data_generation(self, batchTreeIndices):

        respectiveNormalizedTargets = [[t] for t in self.normalizedTargets[batchTreeIndices]]
        targets = np.array(respectiveNormalizedTargets)

        haps = []
        pos = []

        for treeIndex in batchTreeIndices:
            Hfilepath = os.path.join(self.treesDirectory,str(treeIndex) + "_haps.npy")
            Pfilepath = os.path.join(self.treesDirectory,str(treeIndex) + "_pos.npy")
            H = np.load(Hfilepath)
            P = np.load(Pfilepath)
            haps.append(H)
            pos.append(P)

        if(self.realLinePos):
            for p in range(len(pos)):
                pos[p] = pos[p] / self.infoDir["ChromosomeLength"]

        if(self.sortInds):
            for i in range(len(haps)):
                haps[i] = np.transpose(self.sort_min_diff(np.transpose(haps[i])))

        if(self.shuffleInds):
            for i in range(len(haps)):
                haps[i] = self.shuffleIndividuals(haps[i])

        if(self.maxLen != None):
            ##then we're probably padding
            haps,pos = self.pad_HapsPos(haps,pos,
                maxSNPs=self.maxLen,
                frameWidth=self.frameWidth,
                center=self.center)

            pos=np.where(pos == -1.0, self.posPadVal,pos)
            haps=np.where(haps < 1.0, self.ancVal, haps)
            haps=np.where(haps > 1.0, self.padVal, haps)
            haps=np.where(haps == 1.0, self.derVal, haps)
            return [haps,pos], targets


class VCFBatchGenerator(keras.utils.Sequence):
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
            hap=True
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
        self.hap=hap

    def shuffleIndividuals(self,x):
        t = np.arange(x.shape[1])
        np.random.shuffle(t)
        return x[:,t]

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
        GT=self.GT.to_haplotypes()
        if self.hap==True:
            GT=GT[:,::2] #Select only the first of the two diploid chromosomes

        haps,pos=[],[]
        for i in range(len(self.IDs)):
            haps.append(GT[self.IDs[i][0]:self.IDs[i][1]])
            pos.append(self.POS[self.IDs[i][0]:self.IDs[i][1]])


        maxPos = self.POS.max()
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



