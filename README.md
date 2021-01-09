# MatScIE: Material science information extration tool

MatSciE is a tool for information extraction from material science scientific documents. Details of the models used in this paper can be found in this [paper] (https://arxiv.org/pdf/2009.06819.pdf).

## BiLSTM-ELMO-CRF
The final model used in this paper is *Mimicking model* which is a BiLSTM-ELMO-CRF model. The code for this model can be found in the ```bilstm_elmo_crf``` directory. Detailed instruction to train and test the models are present inside the directory.


## BERT Domain Adaptation Model
This model was used as a baseline to compare the results. Its code can be found [here] (https://github.com/xhan77/AdaptaBERT).
We applied the code present in the above repo for IKST dataset. No changes in the existing model was made

### Command to run :-
'''python -W ignore task-tuning.py --data_dir="data/" --bert_model="scibert_scivocab_cased" --output_dir="trained_model_BERT_NER_scibert/" --trained_model_dir="lm_output/" --max_seq_length=256 --do_train --do_eval --do_test --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=30 --warmup_proportion=0.1 --supervised_training --seed=2019'''
