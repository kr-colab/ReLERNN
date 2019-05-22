SIMULATE="../ReLERNN_SIMULATE.py"
TRAIN="../ReLERNN_TRAIN.py"
PREDICT="../ReLERNN_PREDICT.py"
BSCORRECT="../ReLERNN_BSCORRECT.py"
CPU="4"
MU="1e-8"
RTR="1"
DIR="./example_output/"
VCF="./example.vcf"

# Simulate data
python ${SIMULATE} \
    --vcf ${VCF} \
    --projectDir ${DIR} \
    --assumedMu ${MU} \
    --upperRhoThetaRatio ${RTR} \
    --nTrain 1000 \
    --nVali 100 \
    --nTest 100 \
    --nCPU ${CPU}

# Train network
python ${TRAIN} \
    --projectDir ${DIR} \
    --nEpochs 2 \
    --nValSteps 2

# Predict
python ${PREDICT} \
    --vcf ${VCF} \
    --projectDir ${DIR}

# Parametric Bootstrapping
python ${BSCORRECT} \
    --projectDir ${DIR} \
    --nCPU ${CPU} \
    --nSlice 10 \
    --nReps 10
