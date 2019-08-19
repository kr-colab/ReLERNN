# *ReLERNN*
## *Recombination Landscape Estimation using Recurrent Neural Networks*
====================================================================

ReLERNN uses deep learning to infer the genome-wide landscape of recombination from as few as four chromosomes.
This repository contains the code and instructions required to run ReLERNN, and includes example files to ensure everything is working properly. The current manuscript detailing ReLERNN can be found [here](https://www.biorxiv.org/content/early/2019/06/06/662247.full.pdf).

## Recommended installation on linux
Install `tensorflow-gpu` on your system. Directions can be found [here](https://www.tensorflow.org/install/gpu). We recommend using tensorflow version 1.13.1. You will need to install the CUDA toolkit and CuDNN as well as mentioned in the docs above.

Further dependencies for ReLERNN can be installed with pip.
This is done with the following command:

```
$ pip install ReLERNN
```

Alternatively, you can clone directly from github and install via setup.py using the following commands: 

```
$ git clone https://github.com/kern-lab/ReLERNN.git
$ cd ReLERNN
$ python setup.py install
```

It should be as simple as that.

## Testing ReLERNN
An example VCF file (10 haploid chromosomes) and a shell script for running ReLERNN's four modules is located in `$/ReLERNN/examples`.
To test the functionality of ReLERNN simply use the following commands:

```
$ cd examples
$ ./example_pipeline.sh
```

Provided everything worked as planned, `$ReLERNN/examples/example_output/` should be populated with a few directories along with the files: `example.PREDICT.txt` and `example.PREDICT.BSCORRECT.txt`.
The latter is the finalized output file with your recombination rate predictions and estimates of uncertainty.

The above example took 57 seconds to complete on a Xeon machine using four CPUs and one NVIDIA 2070 GPU.
Note that the parameters used for this example were designed only to test the success of the installation, not to make accurate predictions.
Please use the guidelines below for the best results when analyzing real data.
ReLERNN requires the use of a CUDA-Enabled NVIDIA GPU.

## Estimating a recombination landscape using ReLERNN

The ReLERNN pipeline is executed using four commands: `ReLERNN_SIMULATE`, `ReLERNN_TRAIN`, `ReLERNN_PREDICT`, and `ReLERNN_BSCORRECT` (see the [Method flow diagram](./methodFlow.png)).

### Before running ReLERNN
ReLERNN takes as input a VCF file of biallelic variants. It is critical that sites with more than two alleles and sites with missing/masked data are filtered from the VCF before running ReLERNN. Additionally, users should use appropriate QC techniques (filtering low-quality variants, etc.) before running ReLERNN. A BED-formatted accessibility mask may be provided using the `--mask` option. 

If you want to make predictions based on equilibrium simulations, you can skip ahead to executing `ReLERNN_SIMULATE`.
While ReLERNN is generally robust to demographic model misspecification, prediction accuracy may potentially be improved by simulating the training set under a demographic history that accurately matches that of your sample. ReLERNN optionally takes the raw output files from three popular demographic history inference programs ([stairwayplot_v1](https://sites.google.com/site/jpopgen/stairway-plot), [SMC++](https://github.com/popgenmethods/smcpp), and [MSMC](https://github.com/stschiff/msmc)), and simulates a training set under these histories. It is up to the user to perform the proper due diligence to ensure that the population size histories reported by these programs are sound. In our opinion, unless you know exactly how these programs work and you expect your data to represent a history dramatically different from equilibrium, you are better off skipping this step and training ReLERNN on equilibrium simulations. Once you have run one of the demographic history inference programs listed above, you simply provide the raw output file from that program to ReLERNN_SIMULATE using the `--demographicHistory` option.


### Step 1) ReLERNN_SIMULATE
`ReLERNN_SIMULATE` reads your VCF file and splits it by chromosome. The genomic positions to evaluate must be specified by providing a BED file of said positions using the `--genome` argument. Use the `--phased` or `--unphased` flag to specify if your data has been phased (This will be automated shortly). It is required that the VCF file use the extension `.vcf`. The prefix of that file will serve as the prefix used for all output files (e.g. running ReLERNN on the file `population7.vcf` will generate the result file `population7.PREDICT.txt`). Users are required to provide an estimate of the per-base mutation rate for your sample, along with an estimate for generation time (in years). If you previously ran one of the demographic history inference programs listed above, just use the same values that you used for them. This is also where you will point to the output from said program, using `--demographicHistory`. If you are not simulating under an inferred history, simply do not include this option. Importantly, you can also set a value for the maximum recombination rate to be simulated using `--upperRhoThetaRatio`. If you have an a priori estimate for an upper bound to the ratio of rho to theta go ahead and set this here. Keep in mind that higher values will dramatically slow the coalescent simulations. We recommend using the default number of train/test/validation simulation examples, but if you want to simulate more examples, go right ahead. `ReLERNN_SIMULATE` then uses msprime to simulate 100k training examples and 1k validation and test examples. All output files will be generated in subdirectories within the path provided to `--projectDir`. Note: It is required that you use the same projectDir for all four ReLERNN commands. If you want to run ReLERNN of multiple populations/taxa, you can run them independently using a unique projectDir for each.

The complete list of options used in `ReLERNN_SIMULATE` are found below:
```
ReLERNN_SIMULATE -h

usage: ReLERNN_SIMULATE [-h] [-v VCF] [-g GENOME] [-m MASK] [-d OUTDIR]
                        [-n DEM] [-u MU] [-l GENTIME] [-r UPRTR] [-t NCPU]
                        [--phased] [--unphased] [--phaseError PHASEERROR]
                        [--maxWinSize WINSIZEMX] [--maskThresh MASKTHRESH]
                        [--nTrain NTRAIN] [--nVali NVALI] [--nTest NTEST]

optional arguments:
  -h, --help            show this help message and exit
  -v VCF, --vcf VCF     Filtered and QC-checked VCF file. Important: Every row
                        must correspond to a biallelic SNP with no missing
                        data!
  -g GENOME, --genome GENOME
                        BED-formatted (i.e. zero-based) file corresponding to
                        chromosomes and positions to consider
  -m MASK, --mask MASK  BED-formatted file corresponding to inaccessible bases
  -d OUTDIR, --projectDir OUTDIR
                        Directory for all project output. NOTE: the same
                        projectDir must be used for all functions of ReLERNN
  -n DEM, --demographicHistory DEM
                        Output file from either stairwayplot, SMC++, or MSMC
  -u MU, --assumedMu MU
                        Assumed per-base mutation rate
  -l GENTIME, --assumedGenTime GENTIME
                        Assumed generation time (in years)
  -r UPRTR, --upperRhoThetaRatio UPRTR
                        Assumed upper bound for the ratio of rho to theta
  -t NCPU, --nCPU NCPU  Number of CPUs to use
  --phased              VCF file is phased
  --unphased            VCF file is unphased
  --phaseError PHASEERROR
                        Fraction of bases simulated with incorrect phasing
  --maxWinSize WINSIZEMX
                        Max number of sites per window to train on. Important:
                        too many sites causes problems in training (see
                        README)!
  --maskThresh MASKTHRESH
                        Discard windows where >= maskThresh percent of sites
                        are inaccessible
  --nTrain NTRAIN       Number of training examples to simulate
  --nVali NVALI         Number of validation examples to simulate
  --nTest NTEST         Number of test examples to simulate
```


### Step 2) ReLERNN_TRAIN
`ReLERNN_TRAIN` takes the simulations created by `ReLERNN_SIMULATE` and uses them to train a recurrent neural network. Again, we recommend using the defaults for `--nEpochs` and `--nValSteps`, but if you would like to do more training, feel free. To set the GPU to be used for machines with multiple dedicated GPUs use `--gpuID` (e.g. if running an analyis on two populations simultaneously, set `--gpuID 0` for the first population and `--gpuID 1` for the second). `ReLERNN_TRAIN` outputs some basic metrics of the training results for you, generating the figure `$/projectDir/networks/vcfprefix.pdf`.

The complete list of options used in `ReLERNN_TRAIN` are found below:
```
ReLERNN_TRAIN -h

usage: ReLERNN_TRAIN [-h] [-d OUTDIR] [--nEpochs NEPOCHS]
                     [--nValSteps NVALSTEPS] [--gpuID GPUID]

optional arguments:
  -h, --help            show this help message and exit
  -d OUTDIR, --projectDir OUTDIR
                        Directory for all project output. NOTE: the same
                        projectDir must be used for all functions of ReLERNN
  --nEpochs NEPOCHS     Number of epochs to train over
  --nValSteps NVALSTEPS
                        Number of validation steps
  --gpuID GPUID         Identifier specifying which GPU to use
```



### Step 3) ReLERNN_PREDICT
`ReLERNN_PREDICT` now takes the same VCF file you used in `ReLERNN_SIMULATE` and predicts per-base recombination rates in non-overlapping windows across the genome. The output file of predictions will be created as `$/projectDir/vcfprefix.PREDICT.txt`. It is important to note that the window size used for predictions might be different for different chromosomes. A complete list of the window sizes used for each chromosome can be found in third column of `$/projectDir/networks/windowSizes.txt`. If you do not want to estimate confidence intervals around the predictions, you are not finished. Use the optional `--minSites` argument to exclude windows with fewer than the desired number of SNPs.


The complete list of options used in `ReLERNN_PREDICT` are found below:
```
ReLERNN_PREDICT -h

usage: ReLERNN_PREDICT [-h] [-v VCF] [-d OUTDIR] [--minSites MINS]
                       [--gpuID GPUID]

optional arguments:
  -h, --help            show this help message and exit
  -v VCF, --vcf VCF     Filtered and QC-checked VCF file. Important: Every row
                        must correspond to a biallelic SNP with no missing
                        data!
  -d OUTDIR, --projectDir OUTDIR
                        Directory for all project output. NOTE: the same
                        projectDir must be used for all functions of ReLERNN
  --minSites MINS       Minimum number of SNPs in a genomic window required to
                        return a prediction
  --gpuID GPUID         Identifier specifying which GPU to use
```

### Optional Step 4) ReLERNN_BSCORRECT
However, you might want to have an idea of the uncertainty around your predictions. This is where `ReLERNN_BSCORRECT` comes in. `ReLERNN_BSCORRECT` generates 95% confidence intervals around each prediction, and additionally attempts to correct for systematic bias ([see Materials and Methods](https://www.biorxiv.org/content/early/2019/06/06/662247.full.pdf)). It does this by simulated a set of `--nReps` examples at each of `nSlice` recombination rate bins. It then uses the network that was trained in `ReLERNN_TRAIN` and estimates the distribution of predictions around each know recombination rate. The result is both an estimate of uncertainty, and a prediction that has been slightly corrected to account for biases in how the network predicts in this area of parameter space. The resulting file is created as `$/projectDir/vcfprefix.PREDICT.BSCORRECT.txt`, and is formatted similarly to `$/projectDir/vcfprefix.PREDICT.txt`, with the addition of columns for the low and high 95CI bounds.

The complete list of options used in `ReLERNN_BSCORRECT` are found below:
```
ReLERNN_BSCORRECT -h

usage: ReLERNN_BSCORRECT [-h] [-d OUTDIR] [-t NCPU] [--gpuID GPUID]
                         [--nSlice NSLICE] [--nReps NREPS]

optional arguments:
  -h, --help            show this help message and exit
  -d OUTDIR, --projectDir OUTDIR
                        Directory for all project output. NOTE: the same
                        projectDir must be used for all functions of ReLERNN
  -t NCPU, --nCPU NCPU  Number of CPUs to use
  --gpuID GPUID         Identifier specifying which GPU to use
  --nSlice NSLICE       Number of recombination rate bins to simulate over
  --nReps NREPS         Number of simulations per step
```


