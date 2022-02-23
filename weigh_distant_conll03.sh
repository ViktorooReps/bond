export CONLL03_TRAIN_FILE=dataset/data/conll03/distant/train.json
export CONLL03_DEV_FILE=dataset/data/conll03/gold/valid.json
export DATA_FOLDER_PREFIX=dataset/data/conll03/splitdata
export MODEL_FOLDER_PREFIX=model
export WEIGHED_MODEL_FOLDER_NAME=weighed
mkdir -p ${DATA_FOLDER_PREFIX}/${WEIGHED_MODEL_FOLDER_NAME}

export PYTHONPATH=.

## creating splits
#for splits in $(seq 1 1 3); do
#    SPLIT_FOLDER=${DATA_FOLDER_PREFIX}/split-${splits}
#    python crossweigh/split.py --train_files ${CONLL03_TRAIN_FILE} \
#                               --dev_file ${CONLL03_DEV_FILE} \
#                               --output_folder ${SPLIT_FOLDER} \
#                               --schema iob \
#		                           --folds 10
#done
#
## training each split/fold
#for splits in $(seq 1 1 3); do
#    for folds in $(seq 0 1 9); do
#        SPLIT_FOLDER=${DATA_FOLDER_PREFIX}/split-${splits}
#        FOLD_FOLDER=split-${splits}/fold-${folds}
#        python crossweigh/flair_scripts/flair_ner.py --folder_name ${FOLD_FOLDER} \
#                                                     --dev_file ${SPLIT_FOLDER}/dev.txt \
#                                                     --data_folder_prefix ${DATA_FOLDER_PREFIX} \
#                                                     --model_folder_prefix ${MODEL_FOLDER_PREFIX}
#    done
#done

# collecting results and forming a weighted train set.
python crossweigh/collect.py --split_folders ${DATA_FOLDER_PREFIX}/split-* \
                             --train_files $CONLL03_TRAIN_FILE $CONLL03_DEV_FILE \
                             --train_file_schema iob \
                             --output ${DATA_FOLDER_PREFIX}/${WEIGHED_MODEL_FOLDER_NAME}/train.bio

