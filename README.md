# *ReLERNN*
## *Recombination Landscape Estimation using Recurrent Neural Networks*
====================================================================

ReLERNN uses deep learning to infer the genome-wide landscape of recombination from as few as four samples.
This repository contains the code and instructions required to run ReLERNN, and includes example files to ensure everything is working properly.   

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
An example VCF file (10 haploid samples) and a shell script for running ReLERNN's four modules is located in $/ReLERNN/examples.
To test the functionality of ReLERNN simply use the following commands:

```
$ cd examples
$ ./example_pipeline.sh
```

Provided everything worked as planned, $ReLERNN/examples/example_output/ should be populated with a few directories along with the files: example.PREDICT.txt and example.PREDICT.BSCORRECT.txt.
The latter is the finalized output file with your recombination rate predictions and estimates of uncertainty.

The above example took 57 seconds to complete on a Xeon machine using four CPUs and one NVIDIA 2070 GPU.
Note that the parameters used for this example were designed only to test the success of the installation, not to make accurate predictions.
Please use the guidelines below for the best results when analyzing real data.
While it is possible to run ReLERNN without a dedicated GPU, if you do try this, you are going to have a bad time.

## Estimating a recombination landscape using ReLERNN

The ReLERNN pipeline is executed using four commands: `ReLERNN_SIMULATE`, `ReLERNN_TRAIN`, `ReLERNN_PREDICT`, and `ReLERNN_BSCORRECT` (see the [Method flow diagram](./methodFlow.png)).

### Before running ReLERNN
ReLERNN takes as input a phased VCF file of biallelic variants. It is critical that multi-allelic sites and missing/masked data are filtered from the VCF before running ReLERNN. Additionally, users should use appropriate QC techniques (filtering low-quality variants, etc.) before running ReLERNN.

If you want to make predictions based on equilibrium simulations, you can skip ahead to executing `ReLERNN_SIMULATE`.
While ReLERNN is generally robust to demographic model misspecification, you will improve the accuracy of your predictions if you simulate the training set under a demographic history that accurately matches that of your sample. ReLERNN optionally takes the raw output files from three popular demographic history inference programs ([stairwayplot_v1](https://sites.google.com/site/jpopgen/stairway-plot), [SMC++](https://github.com/popgenmethods/smcpp), and [MSMC](https://github.com/stschiff/msmc)), and simulates a training set under these histories. It is up to the user to perform the proper due diligence to ensure that the population size histories reported by these programs are sound. In our opinion, unless you know exactly how these programs work and you expect your data to represent a history dramatically different from equilibrium, you are better off skipping this step and training ReLERNN on equilibrium simulations. Once you have run one of the demographic history inference programs listed above, you simply provide the raw output file from that program to ReLERNN_SIMULATE using the `--demographicHistory` option.


### Step 1) ReLERNN_SIMULATE
`ReLERNN_SIMULATE` reads your VCF file, splits it by chromosome, and then calculates Watterson's theta to arrive at  appropriate simlulation parameters. Users are required to provide an estimate of the per-base mutation rate for your sample, along with an estimate for generation time (in years). If you previously ran one of the demographic history inference programs listed above, just use the same values that you used for them. This is also where you will point to the output from said program, using `--demographicHistory`. If you are not simulating under an inferred history, simply do not include this option. Importantly, you can also set a value for the maximum recombination rate to be simulated using `--upperRhoThetaRatio`. If you have an a priori estimate for an upper bound to the ratio of rho to theta go ahead and set this here. Keep in mind that higher values of recombination will dramatically slow the coalescent simulations. We recommend using the default number of train/test/validation simulation examples, but if you want to simulate more examples, go right ahead. `ReLERNN_SIMULATE` then uses msprime to simulate 100k training examples and 1k validation and test examples. All output files will be generated in subdirectories within the path provided to `--projectDir`. Note: It is required that you use the same projectDir for all four ReLERNN commands. If you want to run ReLERNN of multiple populations/taxa, you can run them independently using a unique projectDir for each.  

```
ReLERNN_SIMULATE -h

usage: ReLERNN_SIMULATE [-h] [-v VCF] [-d OUTDIR] [-n DEM] [-m MU]
                        [-g GENTIME] [-r UPRTR] [--nTrain NTRAIN]
                        [--nVali NVALI] [--nTest NTEST] [-t NCPU]

optional arguments:
  -h, --help            show this help message and exit
  -v VCF, --vcf VCF     Filtered and QC-checked VCF file Note: Every row must
                        correspond to a biallelic SNP with no missing data)
  -d OUTDIR, --projectDir OUTDIR
                        Directory for all project output. NOTE: the same
                        projectDir must be used for all functions of ReLERNN
  -n DEM, --demographicHistory DEM
                        Output file from either stairwayplot, SMC++, or MSMC
  -m MU, --assumedMu MU
                        Assumed per-base mutation rate
  -g GENTIME, --assumedGenTime GENTIME
                        Assumed generation time (in years)
  -r UPRTR, --upperRhoThetaRatio UPRTR
                        Upper bound for the assumed ratio between rho and
                        theta
  --nTrain NTRAIN       Number of training examples to simulate
  --nVali NVALI         Number of validation examples to simulate
  --nTest NTEST         Number of test examples to simulate
  -t NCPU, --nCPU NCPU  Number of CPUs to use
```
