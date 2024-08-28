# EasyECR

## ECRData
- [DataLoader](./easyecr/ecr_data/data_converter/data_converter.py)
  - ECBPlus
  - FCC
  - FCCT
  - GVC
  - WEC
  - ACE2005ENG
  - KBP2015
  - KBP2016ENG
  - KBP2017ENG
  - MAVEN
  - MAVEN-ERE

## Pipelines
### LemmaECR
- sh run.sh 0 easyecr/pipeline/two_n_is_better_than_n2_pipeline.py --config_filename two_n_is_better_than_n2_ecbplus
- sh run.sh 0 easyecr/pipeline/two_n_is_better_than_n2_pipeline.py --config_filename two_n_is_better_than_n2_fcct
- sh run.sh 0 easyecr/pipeline/two_n_is_better_than_n2_pipeline.py --config_filename two_n_is_better_than_n2_gvc
- sh run.sh 0 easyecr/pipeline/two_n_is_better_than_n2_pipeline.py --config_filename two_n_is_better_than_n2_wec
