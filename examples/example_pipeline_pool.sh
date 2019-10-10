SIMULATE="ReLERNN_SIMULATE_POOL"
TRAIN="ReLERNN_TRAIN_POOL"
PREDICT="ReLERNN_PREDICT_POOL"
BSCORRECT="ReLERNN_BSCORRECT"
CPU="4"
MU="1e-8"
RTR="1"
DIR="./example_output_pool/"
POOL="./example.pool"
GENOME="./genome.bed"
MASK="./accessibility_mask.bed"

# Simulate data
${SIMULATE} \
    --pool ${POOL} \
    --sampleDepth 20 \
    --genome ${GENOME} \
    --mask ${MASK} \
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
    --readDepth 20 \
    --maf 0.05 \
    --nEpochs 2 \
    --nValSteps 2 \
    --nCPU ${CPU}

# Predict
${PREDICT} \
    --pool ${POOL} \
    --projectDir ${DIR}

# Parametric Bootstrapping
${BSCORRECT} \
    --projectDir ${DIR} \
    --nCPU ${CPU} \
    --nSlice 10 \
    --nReps 10
