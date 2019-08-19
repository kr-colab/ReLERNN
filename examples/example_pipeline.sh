SIMULATE="ReLERNN_SIMULATE"
TRAIN="ReLERNN_TRAIN"
PREDICT="ReLERNN_PREDICT"
BSCORRECT="ReLERNN_BSCORRECT"
CPU="4"
MU="1e-8"
RTR="1"
DIR="./example_output/"
VCF="./example.vcf"
GENOME="./genome.bed"
MASK="./accessibility_mask.bed"

# Simulate data
${SIMULATE} \
    --vcf ${VCF} \
    --genome ${GENOME} \
    --mask ${MASK} \
    --phased \
    --projectDir ${DIR} \
    --assumedMu ${MU} \
    --upperRhoThetaRatio ${RTR} \
    --nTrain 1000 \
    --nVali 100 \
    --nTest 100 \
    --nCPU ${CPU}

# Train network
${TRAIN} \
    --projectDir ${DIR} \
    --nEpochs 2 \
    --nValSteps 2

# Predict
${PREDICT} \
    --vcf ${VCF} \
    --projectDir ${DIR}

# Parametric Bootstrapping
${BSCORRECT} \
    --projectDir ${DIR} \
    --nCPU ${CPU} \
    --nSlice 10 \
    --nReps 10
