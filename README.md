# [ICCV2025] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning

by Zhengxuan Wei*, Jiajin Tang*, Sibei Yang†

*Equal contribution; †Corresponding Author


[![arXiv:2510.19622](https://img.shields.io/badge/arXiv-2510.19622-red)](https://arxiv.org/abs/2510.19622)

----------
## Abstract

Existing Moment Retrieval methods face three critical bottlenecks: (1) data scarcity forces models into shallow keyword-feature associations; (2) boundary ambiguity in transition regions between adjacent events; (3) insufficient discrimination of fine-grained semantics (e.g., distinguishing "kicking" vs. "throwing" a ball). In this paper, we propose a zero-external-dependency Augmented Moment Retrieval framework, AMR, designed to overcome local optima caused by insufficient data annotations and the lack of robust boundary and semantic discrimination capabilities. AMR is built upon two key insights: (1) it resolves ambiguous boundary information and semantic confusion in existing annotations without additional data (avoiding costly manual labeling), and (2) it preserves boundary and semantic discriminative capabilities enhanced by training while generalizing to real-world scenarios, significantly improving performance. Furthermore, we propose a two-stage training framework with cold-start and distillation adaptation. The cold-start stage employs curriculum learning on augmented data to build foundational boundary/semantic awareness. The distillation stage introduces dual query sets: Original Queries maintain DETR-based localization using frozen Base Queries from the cold-start model, while Active Queries dynamically adapt to real-data distributions. A cross-stage distillation loss enforces consistency between Original and Base Queries, preventing knowledge forgetting while enabling real-world generalization. Experiments on multiple benchmarks show that AMR achieves improved performance over prior state-of-the-art approaches.

----------
## Framework
<p align="center">
  <img src="assets/framework.png" width="700"/>
</p>

----------

## Prerequisites

### 0. Clone this repository

```
git clone https://github.com/SooLab/AMR.git
cd AMR
```

### 1. Prepare datasets
#### QVHighlights

We use video features (CLIP and SlowFast) and text features (CLIP) as inputs. For CLIP, we utilize the features extracted by [R2-Tuning](https://github.com/yeliudev/R2-Tuning) (from the last four layers), but we retain only the `[CLS]` token per frame to ensure efficiency. You can download our prepared feature files from [qvhighlights\_features](https://drive.google.com/drive/folders/1rRVID6OO5arVR1vL5SP5fcCFAJ35B-IK?usp=sharing) and unzip them to your data root directory.


### 2. Install dependencies

For Anaconda setup, refer to the official [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr).

----------

## QVHighlights

### Data Augmentation
We provide an offline data augmentation script `utils/augment_data.py`. Before running the script, ensure you update the `data_root` variable to point to your data root directory. Run the following command to generate augmented training data:

```bash
python utils/augment_data.py
```

### Training - Stage 1

Update `feat_root` in `amr/scripts/train_stage1.sh` to the path where you saved the features, then run:

```bash
bash amr/scripts/train_stage1.sh  
```

### Training - Stage 2

Update `feat_root` in `amr/scripts/train_stage2.sh` to the path where you saved the features. Also, set the `resume` parameter to point to the checkpoint saved from Stage 1 (e.g., `test_amr/{direc}/model_e0039.ckpt`). Then run:

```bash
bash amr/scripts/train_stage2.sh  
```


### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
bash amr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash amr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

Replace `{direc}` with the path to your saved checkpoint. For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## Citation

If you find this repository useful, please cite our work:

```
@inproceedings{wei2025augmenting,
  title={Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning},
  author={Wei, Zhengxuan and Tang, Jiajin and Yang, Sibei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={3401--3412},
  year={2025}
}
```

----------

## License

The annotation files and parts of the implementation are borrowed from Moment-DETR and TR-DETR. Consequently, our code is also released under the [MIT License](https://opensource.org/licenses/MIT).


