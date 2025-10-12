This Directory contains implementation of domainbed on PACS dataset for ERM, IRM, and Group-DRO. we have used leave one out policy to see effective domain generalization and also analyzing which domain is hardest to generalize.
the following commands are used to train data 

!====== Training Models  ======!
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm ERM --dataset PACS  --test_env 0 --output_dir=./domainbed/results/ERM_RESULTS_0 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm IRM --dataset PACS  --test_env 0 --output_dir=./domainbed/results/IRM_RESULTS_0 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GroupDRO --dataset PACS  --test_env 0 --output_dir=./domainbed/results/GroupDRO_RESULTS_0 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint

python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm ERM --dataset PACS  --test_env 1 --output_dir=./domainbed/results/ERM_RESULTS_1 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm IRM --dataset PACS  --test_env 1 --output_dir=./domainbed/results/IRM_RESULTS_1 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GroupDRO --dataset PACS  --test_env 1 --output_dir=./domainbed/results/GroupDRO_RESULTS_1 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint

python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm ERM --dataset PACS  --test_env 2 --output_dir=./domainbed/results/ERM_RESULTS_2 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm IRM --dataset PACS  --test_env 2 --output_dir=./domainbed/results/IRM_RESULTS_2 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GroupDRO --dataset PACS  --test_env 2 --output_dir=./domainbed/results/GroupDRO_RESULTS_2 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint


python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm ERM --dataset PACS  --test_env 3 --output_dir=./domainbed/results/ERM_RESULTS_3 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm IRM --dataset PACS  --test_env 3 --output_dir=./domainbed/results/IRM_RESULTS_3 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint
python -m domainbed.scripts.train --data_dir=./domainbed/data --algorithm GroupDRO --dataset PACS  --test_env 3 --output_dir=./domainbed/results/GroupDRO_RESULTS_3 --hparams '{\"batch_size\":64,\"lr\":5e-5}' --save_model_every_checkpoint


the saved model checkpoints are saved to https://huggingface.co/Aizelsheikh/erm_resnet18_pacs and can be downloaded for inference

DESCRIPTION OF FILES
each file is saved with _0, _1, _2, _3 format e.g RESULTS_ERM_0.jsonl etc..... 
here 0: art_painting
     1: cartoon
     2: photo
     3: sketch
the number after underscore reflects the one leave out domain for testing.
