dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=test_amr
exp_id=exp

######## data paths
train_path=/data1/wzx/AMR/data/highlight_train_with_neg_spans.jsonl
eval_path=data/highlight_val_release.jsonl
eval_split_name=val

######## setup video+text features
feat_root=/data1/wzx/datasets/qvhighlights_feature
resume='Your_pretrained_model_path/model_e0039.ckpt'


# video features
v_feat_dim=0
v_feat_dirs=()
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=(${feat_root}/slowfast_features)
  (( v_feat_dim += 2304 ))  # double brackets for arithmetic op, no need to use ${v_feat_dim}
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=(${feat_root}/clip_b32_vid_k4)
  (( v_feat_dim += 3072 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir=${feat_root}/clip_b32_txt_k4
  t_feat_dim=2048
else
  echo "Wrong arg for t_feat_type."
  exit 1
fi

#### training
bsz=32
lr_drop=40
lr=0.0001
n_epoch=100
lw_saliency=1.0
seed=42
VTC_loss_coef=0.3
CTC_loss_coef=0.5
label_loss_coef=4
disc_loss_coef=0.5
dill_loss_coef=0.5
stage="distill"


PYTHONPATH=$PYTHONPATH:. python amr/train.py \
--seed $seed \
--label_loss_coef $label_loss_coef \
--VTC_loss_coef $VTC_loss_coef \
--CTC_loss_coef $CTC_loss_coef \
--dset_name ${dset_name} \
--ctx_mode ${ctx_mode} \
--train_path ${train_path} \
--eval_path ${eval_path} \
--eval_split_name ${eval_split_name} \
--v_feat_dirs ${v_feat_dirs[@]} \
--v_feat_dim ${v_feat_dim} \
--t_feat_dir ${t_feat_dir} \
--t_feat_dim ${t_feat_dim} \
--bsz ${bsz} \
--results_root ${results_root} \
--exp_id ${exp_id} \
--lr ${lr} \
--n_epoch ${n_epoch} \
--lw_saliency ${lw_saliency} \
--lr_drop ${lr_drop} \
--disc_loss_coef ${disc_loss_coef} \
--dill_loss_coef ${dill_loss_coef} \
--stage ${stage} \
${@:1}
