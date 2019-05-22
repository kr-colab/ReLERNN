
'''
Authors: Jared Galloway, Jeff Adrion
'''

from ReLERNN.imports import *
from ReLERNN.simulator import *
from ReLERNN.sequenceBatchGenerator import *

def relu(x):
    return max(0,x)

#-------------------------------------------------------------------------------------------

def zscoreTargets(self):
    norm = self.targetNormalization
    nTargets = copy.deepcopy(self.infoDir['y'])
    if(norm == 'zscore'):
        tar_mean = np.mean(nTargets,axis=0)
        tar_sd = np.std(nTargets,axis=0)
        nTargets -= tar_mean
        nTargets = np.divide(nTargets,tar_sd,out=np.zeros_like(nTargets),where=tar_sd!=0)

#-------------------------------------------------------------------------------------------

def load_and_predictVCF(VCFGenerator,
            resultsFile=None,
            network=None,
            gpuID = 0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: no pretrained network found!")
        sys.exit(1)

    x,chrom,win,info,nSNPs = VCFGenerator.__getitem__(0)
    predictions = model.predict(x)

    u=np.mean(info["rho"])
    sd=np.std(info["rho"])
    with open(resultsFile, "w") as fOUT:
        ct=0
        fOUT.write("%s\t%s\t%s\t%s\n" %("chrom","start","end","recombRate"))
        for i in range(len(predictions)):
            if nSNPs[i] >= 20:
                fOUT.write("%s\t%s\t%s\t%s\n" %(chrom,ct,ct+win,relu(sd*predictions[i][0]+u)))
            ct+=win

    return None

#-------------------------------------------------------------------------------------------

def runModels(ModelFuncPointer,
            ModelName,
            TrainDir,
            TrainGenerator,
            ValidationGenerator,
            TestGenerator,
            resultsFile=None,
            numEpochs=10,
            epochSteps=100,
            validationSteps=1,
            outputNetwork=None,
            gpuID = 0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    if(resultsFile == None):

        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    x,y = TrainGenerator.__getitem__(0)
    model = ModelFuncPointer(x,y)

    history = model.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        use_multiprocessing=False
        )

    x,y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)

    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName

    if(outputNetwork != None):
        ##serialize model to JSON
        model_json = model.to_json()
        with open(outputNetwork[0], "w") as json_file:
            json_file.write(model_json)
        ##serialize weights to HDF5
        model.save_weights(outputNetwork[1])

    print("results written to: ",resultsFile)
    pickle.dump(history.history, open( resultsFile, "wb" ))

    return None

#-------------------------------------------------------------------------------------------

def indicesGenerator(batchSize,numReps):
    '''
    Generate indices randomly from range (0,numReps) in batches of size batchSize
    without replacement.

    This is for the batch generator to randomly choose trees from a directory
    but make sure
    '''
    availableIndices = np.arange(numReps)
    np.random.shuffle(availableIndices)
    ci = 0
    while 1:
        if((ci+batchSize) > numReps):
            ci = 0
            np.random.shuffle(availableIndices)
        batchIndices = availableIndices[ci:ci+batchSize]
        ci = ci+batchSize

        yield batchIndices

#-------------------------------------------------------------------------------------------

def getHapsPosLabels(direc,simulator,shuffle=False):
    '''
    loops through a trees directory created by the data generator class
    and returns the repsective genotype matrices, positions, and labels
    '''
    haps = []
    positions = []
    infoFilename = os.path.join(direc,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    #how many trees files are in this directory.
    li = os.listdir(direc)
    numReps = len(li) - 1   #minus one for the 'info.p' file

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(direc,filename)
        treeSequence = msp.load(filepath)
        haps.append(treeSequence.genotype_matrix())
        positions.append(np.array([s.position for s in treeSequence.sites()]))


    haps = np.array(haps)
    positions = np.array(positions)

    return haps,positions,labels

#-------------------------------------------------------------------------------------------

def simplifyTreeSequenceOnSubSampleSet_stub(ts,numSamples):
    '''
    This function should take in a tree sequence, generate
    a subset the size of numSamples, and return the tree sequence simplified on
    that subset of individuals
    '''

    ts = ts.simplify() #is this neccessary
    inds = [ind.id for ind in ts.individuals()]
    sample_subset = np.sort(np.random.choice(inds,sample_size,replace=False))
    sample_nodes = []
    for i in sample_subset:
        ind = ts.individual(i)
        sample_nodes.append(ind.nodes[0])
        sample_nodes.append(ind.nodes[1])

    ts = ts.simplify(sample_nodes)

    return ts

#-------------------------------------------------------------------------------------------

def shuffleIndividuals(x):
    t = np.arange(x.shape[1])
    np.random.shuffle(t)
    return x[:,t]

#-------------------------------------------------------------------------------------------

def sort_min_diff(amat):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''

    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

#-------------------------------------------------------------------------------------------

def mutateTrees(treesDirec,outputDirec,muLow,muHigh,numMutsPerTree=1,simulator="msprime"):
    '''
    read in .trees files from treesDirec, mutate that tree numMuts seperate times
    using a mutation rate pulled from a uniform dirstribution between muLow and muHigh

    also, re-write the labels file to reflect.
    '''
    if(numMutsPerTree > 1):
        assert(treesDirec != outputDirec)

    if not os.path.exists(outputDirec):
        print("directory '",outputDirec,"' does not exist, creating it")
        os.makedirs(outputDirec)

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))
    labels = infoDict["y"]

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(treesDirec,filename)
        treeSequence = msp.load(filepath)
        blankTreeSequence = msp.mutate(treeSequence,0)
        rho = labels[i]
        for mut in range(numMuts):
            simNum = (i*numMuts) + mut
            simFileName = os.path.join(outputDirec,str(simNum)+".trees")
            mutationRate = np.random.uniform(muLow,muHigh)
            mutatedTreeSequence = msp.mutate(blankTreeSequence,mutationRate)
            mutatedTreeSequence.dump(simFileName)
            newMaxSegSites = max(newMaxSegSites,mutatedTreeSequence.num_sites)
            newLabels.append(rho)

    infoCopy = copy.deepcopy(infoDict)
    infoCopy["maxSegSites"] = newMaxSeqSites
    if(numMutsPerTree > 1):
        infoCopy["y"] = np.array(newLabels,dtype="float32")
        infoCopy["numReps"] = numReps * numMuts
    outInfoFilename = os.path.join(outputDirec,"info.p")
    pickle.dump(infocopy,open(outInfoFilename,"wb"))

    return None

#-------------------------------------------------------------------------------------------

def segSitesStats(treesDirec):
    '''
    DEPRICATED
    '''

    infoFilename = os.path.join(treesDirec,"info.p")
    infoDict = pickle.load(open(infoFilename,"rb"))

    newLabels = []
    newMaxSegSites = 0

    #how many trees files are in this directory.
    li = os.listdir(treesDirec)
    numReps = len(li) - 1   #minus one for the 'labels.txt' file

    segSites = []

    for i in range(numReps):
        filename = str(i) + ".trees"
        filepath = os.path.join(treesDirec,filename)
        treeSequence = msp.load(filepath)
        segSites.append(treeSequence.num_sites)

    return segSites

#-------------------------------------------------------------------------------------------

def mae(x,y):
    '''
    Compute mean absolute error between predictions and targets

    float[],float[] -> float
    '''
    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += abs(x[i] - y[i])
    return summ/length

#-------------------------------------------------------------------------------------------

def mse(x,y):
    '''
    Compute mean squared error between predictions and targets

    float[],float[] -> float
    '''

    assert(len(x) == len(y))
    summ = 0.0
    length = len(x)
    for i in range(length):
        summ += (x[i] - y[i])**2
    return summ/length

#-------------------------------------------------------------------------------------------

def plotResults(resultsFile,saveas):

    '''
    plotting code for testing a model on simulation.
    using the resulting pickle file on a training run (resultsFile).
    This function plots the results of the final test set predictions,
    as well as validation loss as a function of Epochs during training.

    str,str -> None

    '''

    plt.rc('font', family='serif', serif='Times')
    plt.rc('xtick', labelsize=6)
    plt.rc('ytick', labelsize=6)
    plt.rc('axes', labelsize=6)

    results = pickle.load(open( resultsFile , "rb" ))

    fig,axes = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)

    predictions = np.array([float(Y) for Y in results["predictions"]])
    realValues = np.array([float(X) for X in results["Y_test"]])

    r_2 = round((np.corrcoef(predictions,realValues)[0,1])**2,5)

    mae_0 = round(mae(realValues,predictions),4)
    mse_0 = round(mse(realValues,predictions),4)
    labels = "$R^{2} = $"+str(r_2)+"\n"+"$mae = $" + str(mae_0)+" | "+"$mse = $" + str(mse_0)

    axes[0].scatter(realValues,predictions,marker = "o", color = 'tab:purple',s=5.0,alpha=0.6)

    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),  # min of both axes
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),  # max of both axes
    ]
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    axes[0].set_title(results["name"]+"\n"+labels,fontsize=6)

    lossRowIndex = 1
    axes[1].plot(results["loss"],label = "mae loss",color='tab:cyan')
    axes[1].plot(results["val_loss"], label= "mae validation loss",color='tab:pink')

    #axes[1].plot(results["mean_squared_error"],label = "mse loss",color='tab:green')
    #axes[1].plot(results["val_mean_squared_error"], label= "mse validation loss",color='tab:olive')

    axes[1].legend(frameon = False,fontsize = 6)
    axes[1].set_ylabel("mse")

    axes[0].set_ylabel(str(len(predictions))+" msprime predictions")
    axes[0].set_xlabel(str(len(realValues))+" msprime real values")
    fig.subplots_adjust(left=.15, bottom=.16, right=.85, top=.92,hspace = 0.5,wspace=0.4)
    height = 7.00
    width = 7.00

    axes[0].grid()
    fig.set_size_inches(height, width)
    fig.savefig(saveas)

#-------------------------------------------------------------------------------------------

def getMeanSDMax(trainDir):
    '''
    get the mean and standard deviation of rho from training set

    str -> int,int,int

    '''
    info = pickle.load(open(trainDir+"/info.p","rb"))
    rho = info["rho"]
    segSites = info["segSites"]
    tar_mean = np.mean(rho,axis=0)
    tar_sd = np.std(rho,axis=0)
    return tar_mean,tar_sd,max(segSites)

#-------------------------------------------------------------------------------------------

def unNormalize(mean,sd,data):
    '''
    un-zcore-ify. do the inverse to get real value predictions

    float,float,float[] -> float[]
    '''

    data *= sd
    data += mean  ##comment this line out for GRU_TUNED84_RELU
    return data

#-------------------------------------------------------------------------------------------

def ParametricBootStrap(simParameters,
                        batchParameters,
                        trainDir,
                        network=None,
                        slices=1000,
                        repsPerSlice=1000,
                        gpuID=0,
                        tempDir="./Temp",
                        out="./ParametricBootstrap.p",
                        nCPU=1):


    '''
    This Function is for understanding network confidense
    over a range of rho, using a parametric bootstrap.

    SIDE NOTE: This will create a "temp" directory for filling
    writing and re-writing the test sets.
    after, it will destroy the tempDir.

    The basic idea being that we take a trained network,
    and iteritevly create test sets of simulation at steps which increase
    between fixed ranges of Rho.

    This function will output a pickle file containing
    a dictionary where the first

    This function will output a pickle file containing
    a dictionary where the ["rho"] key contains the slices
    between the values of rho where we simulate a test set,
    and test the trained model.

    The rest of the ket:value pairs in the dictionary contain
    the quartile information at each slice position for the
    distribution of test results
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    # load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: no pretrained network found!")

    if not os.path.exists(tempDir):
        os.makedirs(tempDir)

    priorLowsRho = simParameters['priorLowsRho']
    priorHighsRho = simParameters['priorHighsRho']

    rhoDiff = (priorHighsRho - priorLowsRho)/slices
    IQR = {"rho":[],"Min":[],"CI95LO":[],"Q1":[],"Q2":[],"Q3":[],"CI95HI":[],"Max":[]}
    rho = [(priorLowsRho+(rhoDiff*i)) for i in range(slices)]
    IQR["rho"] = rho

    mean,sd,pad = getMeanSDMax(trainDir)

    for idx,r in enumerate(rho):
        print("Simulating slice ",idx," out of ",slices)

        params = copy.deepcopy(simParameters)
        params["priorLowsRho"] = r
        params["priorHighsRho"] = r
        params.pop("bn", None)
        simulator = Simulator(**params)

        simulator.simulateAndProduceTrees(numReps=repsPerSlice,
                                            direc=tempDir,
                                            simulator="msprime",
                                            nProc=nCPU)

        batch_params = copy.deepcopy(batchParameters)
        batch_params['treesDirectory'] = tempDir
        batch_params['batchSize'] = repsPerSlice
        batch_params['shuffleExamples'] = False
        batchGenerator= SequenceBatchGenerator(**batch_params)

        x,y = batchGenerator.__getitem__(0)
        predictions = unNormalize(mean,sd,model.predict(x))
        predictions = [p[0] for p in predictions]

        minP,maxP = min(predictions),max(predictions)
        quartiles = np.percentile(predictions,[2.5,25,50,75,97.5])

        IQR["Min"].append(relu(minP))
        IQR["Max"].append(relu(maxP))
        IQR["CI95LO"].append(relu(quartiles[0]))
        IQR["Q1"].append(relu(quartiles[1]))
        IQR["Q2"].append(relu(quartiles[2]))
        IQR["Q3"].append(relu(quartiles[3]))
        IQR["CI95HI"].append(relu(quartiles[4]))

        del simulator
        del batchGenerator

    pickle.dump(IQR,open(out,"wb"))

    return rho,IQR


def plotParametricBootstrap(results,saveas):

    '''
    Use the location of "out" paramerter to parametric bootstrap
    as input to plot the results of said para-boot
    '''

    stats = pickle.load(open(results,'rb'))
    x = stats["rho"]

    fig, ax = plt.subplots()


    for i,s in enumerate(stats):
        if(i == 0):
            continue

        ax.plot(x,stats[s])

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    #print("finished")
    fig.savefig(saveas)

    return None














