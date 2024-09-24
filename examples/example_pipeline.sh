SIMULATE="ReLERNN_SIMULATE"
TRAIN="ReLERNN_TRAIN"
PREDICT="ReLERNN_PREDICT"
BSCORRECT="ReLERNN_BSCORRECT"
SEED="42"
MU="1e-8"
URTR="1"
DIR="./example_output/"
VCF="./example.vcf"
GENOME="./genome.bed"
MASK="./accessibility_mask.bed"

# Simulate data
${SIMULATE} \
    --vcf ${VCF} \
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
    --seed ${SEED}

# Predict
${PREDICT} \
    --vcf ${VCF} \
    --projectDir ${DIR} \
    --seed ${SEED}

# Parametric Bootstrapping
${BSCORRECT} \
    --projectDir ${DIR} \
    --nSlice 2 \
    --nReps 2 \
    --seed ${SEED}
