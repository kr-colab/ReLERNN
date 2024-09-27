SIMULATE="ReLERNN_SIMULATE_POOL"
TRAIN="ReLERNN_TRAIN_POOL"
PREDICT="ReLERNN_PREDICT_POOL"
BSCORRECT="ReLERNN_BSCORRECT"
SEED="42"
MU="1e-8"
URTR="1"
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
    --upperRhoThetaRatio ${URTR} \
    --nTrain 13000 \
    --nVali 2000 \
    --nTest 100 \
    --seed ${SEED}

# Train network
${TRAIN} \
    --projectDir ${DIR} \
    --readDepth 20 \
    --maf 0.05 \
    --seed ${SEED}

# Predict
${PREDICT} \
    --pool ${POOL} \
    --projectDir ${DIR} \
    --seed ${SEED}

# Parametric Bootstrapping
${BSCORRECT} \
    --projectDir ${DIR} \
    --seed ${SEED}
