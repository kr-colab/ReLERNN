
'''
Authors: Jared Galloway, Jeff Adrion
'''

from ReLERNN.imports import *

#-------------------------------------------------------------------------------------------

def assign_task(mpID, task_q, nProcs):
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

#-------------------------------------------------------------------------------------------

def create_procs(nProcs, task_q, result_q, params, worker):
    pids = []
    for _ in range(nProcs):
        p = mp.Process(target=worker, args=(task_q, result_q, params))
        p.daemon = True
        p.start()
        pids.append(p)
    return pids

#-------------------------------------------------------------------------------------------

def get_corrected_index(L,N):
    idx,outN="",""
    dist=float("inf")
    for i in range(len(L)):
        D=abs(N-L[i])
        if D < dist:
            idx=i
            outN=L[i]
            dist=D
    return [idx,outN]

#-------------------------------------------------------------------------------------------

def get_corrected(rate,bs):
    idx=get_corrected_index(bs["Q2"],rate)
    CI95LO=bs["CI95LO"][idx[0]]
    CI95HI=bs["CI95HI"][idx[0]]
    cRATE=relu(rate+(bs["rho"][idx[0]]-idx[1]))
    ciHI=relu(cRATE+(CI95HI-idx[1]))
    ciLO=relu(cRATE+(CI95LO-idx[1]))
    return [cRATE,ciLO,ciHI]

#-------------------------------------------------------------------------------------------

def get_index(pos, winSize):
    y=snps_per_win(pos,winSize)
    st=0
    indices=[]
    for i in range(len(y)):
        indices.append([st,st+y[i]])
        st+=y[i]
    return indices

#-------------------------------------------------------------------------------------------

def snps_per_win(pos, window_size):
    bins = np.arange(1, pos.max()+window_size, window_size) #use 1-based coordinates, per VCF standard
    y,x = np.histogram(pos,bins=bins)
    return y

#-------------------------------------------------------------------------------------------

def find_win_size(winSize, pos, winSizeMx):
    snpsWin=snps_per_win(pos,winSize)
    mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
    if mx > winSizeMx:
        return [-1]
    elif mx < winSizeMx:
        return [1]
    else:
        return [winSize,mn,u,mx,len(snpsWin)]

#-------------------------------------------------------------------------------------------

def force_win_size(winSize, pos):
    snpsWin=snps_per_win(pos,winSize)
    mn,u,mx = snpsWin.min(), int(snpsWin.mean()), snpsWin.max()
    return [winSize,mn,u,mx,len(snpsWin)]

#-------------------------------------------------------------------------------------------

def maskStats(wins, last_win, mask, maxLen):
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

#-------------------------------------------------------------------------------------------

def check_demHist(path):
    fTypeFlag = -9
    with open(path, "r") as fIN:
        for line in fIN:
            if line.startswith("mutation_per_site"):
                fTypeFlag = 1
                break
            if line.startswith("label"):
                fTypeFlag = 2
                break
            if line.startswith("time_index"):
                fTypeFlag = 3
                break
    return fTypeFlag

#-------------------------------------------------------------------------------------------

def convert_msmc_output(results_file, mutation_rate, generation_time):
   """
   This function converts the output from msmc into a csv the will be read in for
   plotting comparison.

   MSMC outputs times and rates scaled by the mutation rate per basepair per generation.
   First, scaled times are given in units of the per-generation mutation rate.
   This means that in order to convert scaled times to generations,
   divide them by the mutation rate. In humans, we used mu=1e-8 per basepair per generation.
   To convert generations into years, multiply by the generation time, for which we used 10 years.

   To get population sizes out of coalescence rates, first take the inverse of the coalescence rate,
   scaledPopSize = 1 / lambda00. Then divide this scaled population size by 2*mu
   """
   outfile = results_file+".csv"
   out_fp = open(outfile, "w")
   in_fp = open(results_file, "r")
   header = in_fp.readline()
   out_fp.write("label,x,y\n")
   for line in in_fp:
       result = line.split()
       time = float(result[1])
       time_generation = time / mutation_rate
       time_years = time_generation * generation_time
       lambda00 = float(result[3])
       scaled_pop_size = 1 / lambda00
       size = scaled_pop_size / (2*mutation_rate)
       out_fp.write(f"pop0,{time_years},{size}\n")
   out_fp.close
   return None

#-------------------------------------------------------------------------------------------

def convert_demHist(path, nSamps, gen, fType, mu):
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
    ## Convert MSMC to similar format to smc++
    if fType == 3:
        convert_msmc_output(path, mu, gen)
        path+=".csv"
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

#-------------------------------------------------------------------------------------------

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
            minS = 50,
            gpuID = 0,
            hotspots = False):

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

    if hotspots:
        with open(resultsFile, "w") as fOUT:
            ct=0
            fOUT.write("%s\t%s\t%s\t%s\t%s\n" %("chrom","start","end","nSites","hotspot"))
            for i in range(len(predictions)):
                if nSNPs[i] >= minS:
                    fOUT.write("%s\t%s\t%s\t%s\t%s\n" %(chrom,ct,ct+win,nSNPs[i],predictions[i][0]))
                ct+=win
    else:
        u=np.mean(info["rho"])
        sd=np.std(info["rho"])
        last = int(os.path.basename(resultsFile).split(".")[0].split("-")[-1])
        with open(resultsFile, "w") as fOUT:
            ct=0
            fOUT.write("%s\t%s\t%s\t%s\t%s\n" %("chrom","start","end","nSites","recombRate"))
            for i in range(len(predictions)):
                if nSNPs[i] >= minS:
                    fOUT.write("%s\t%s\t%s\t%s\t%s\n" %(chrom,ct,min(ct+win,last),nSNPs[i],relu(sd*predictions[i][0]+u)))
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
            network=None,
            nCPU = 1,
            gpuID = 0):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuID)

    if(resultsFile == None):

        resultsFilename = os.path.basename(trainFile)[:-4] + ".p"
        resultsFile = os.path.join("./results/",resultsFilename)

    x,y = TrainGenerator.__getitem__(0)
    model = ModelFuncPointer(x,y)

    # Early stopping and saving the best weights
    callbacks_list = [
            EarlyStopping(
                monitor='val_loss',
                verbose=1,
                min_delta=0.01,
                patience=100),
            ModelCheckpoint(
                filepath=network[1],
                monitor='val_loss',
                save_best_only=True)
            ]

    history = model.fit_generator(TrainGenerator,
        steps_per_epoch= epochSteps,
        epochs=numEpochs,
        validation_data=ValidationGenerator,
        validation_steps=validationSteps,
        use_multiprocessing=True,
        callbacks=callbacks_list,
        max_queue_size=nCPU,
        workers=nCPU,
        )

    # Write the network
    if(network != None):
        ##serialize model to JSON
        model_json = model.to_json()
        with open(network[0], "w") as json_file:
            json_file.write(model_json)

    # Load json and create model
    if(network != None):
        jsonFILE = open(network[0],"r")
        loadedModel = jsonFILE.read()
        jsonFILE.close()
        model=model_from_json(loadedModel)
        model.load_weights(network[1])
    else:
        print("Error: model and weights not loaded")
        sys.exit(1)

    x,y = TestGenerator.__getitem__(0)
    predictions = model.predict(x)

    history.history['loss'] = np.array(history.history['loss'])
    history.history['val_loss'] = np.array(history.history['val_loss'])
    history.history['predictions'] = np.array(predictions)
    history.history['Y_test'] = np.array(y)
    history.history['name'] = ModelName

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

    fig.savefig(saveas)

    return None

