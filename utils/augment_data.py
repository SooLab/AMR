import os
import json
import random
import numpy as np
from tqdm import tqdm
    
data_root = '/data1/wzx/datasets/qvhighlights_feature/'


train_data_path = 'data/highlight_train_with_neg_spans.jsonl'
save_path = 'data/highlight_train_aug.jsonl'
clip_feat_path = data_root + "clip_b32_vid_k4"
sf_feat_path = data_root + "slowfast_features"
aug_clip_feat_save_path = data_root + "clip_b32_vid_k4_aug"
aug_sf_feat_save_path = data_root + "slowfast_features_aug"

if not os.path.exists(aug_clip_feat_save_path):
    os.makedirs(aug_clip_feat_save_path)
if not os.path.exists(aug_sf_feat_save_path):
    os.makedirs(aug_sf_feat_save_path)

with open(train_data_path, 'r') as f:
    train_data = [json.loads(line) for line in f]
data_num = len(train_data)
iter_num = 10

augmented_data = []
for i in range(iter_num):
    print(f'Augmenting {i+1}/{iter_num}...')
    for j in tqdm(range(data_num)):
        try:
            aug_idx = random.randint(0, data_num-1)
            while aug_idx == j:
                aug_idx = random.randint(0, data_num-1)
            ori_data = train_data[j]            
            aug_data = train_data[aug_idx]
            new_data = {}
            new_data['qid'] = aug_data['qid']
            new_data['query'] = aug_data['query']
            dur = ori_data['duration']
            new_data['duration'] = dur
            ori_vid = ori_data['vid']
            aug_vid = aug_data['vid']
            new_data['vid'] = f'{ori_vid}_aug_{i}'
            ori_clip_feat_path = clip_feat_path+"/"+f"{ori_vid}.npy"
            ori_sf_feat_path = sf_feat_path+"/"+f"{ori_vid}.npz"
            aug_clip_feat_path = clip_feat_path+"/"+f"{aug_vid}.npy"
            aug_sf_feat_path = sf_feat_path+"/"+f"{aug_vid}.npz"
            ori_clip_feat = np.load(ori_clip_feat_path)
            ori_sf_feat = np.load(ori_sf_feat_path)["features"]
            aug_clip_feat = np.load(aug_clip_feat_path)
            aug_sf_feat = np.load(aug_sf_feat_path)["features"]
            aug_true_windows = aug_data['relevant_windows']
            aug_wrong_windows = aug_data['wrong_spans']
            aug_windows = aug_true_windows + aug_wrong_windows
            wrong_flag = [0] * len(aug_true_windows) + [1] * len(aug_wrong_windows)
            combined = list(zip(aug_windows, wrong_flag))
            sorted_combined = sorted(combined, key=lambda x: x[0][0])
            aug_windows, wrong_flag = zip(*sorted_combined)
            aug_windows = list(aug_windows)
            wrong_flag = list(wrong_flag)
            num_aug_windows = len(aug_windows)
            aug_scores = aug_data['saliency_scores']
            new_clip_ids = []
            new_saliency_scores = []
            new_relevant_windows = []
            new_clip_feat = ori_clip_feat.copy()
            new_sf_feat = ori_sf_feat.copy()
            begin = 0
            for k in range(num_aug_windows):
                len_window = (aug_windows[k][1] - aug_windows[k][0])//2
                if wrong_flag[k] == 0:
                    new_scores = aug_scores[:len_window]
                    aug_scores = aug_scores[len_window:]
                if begin > dur//2-len_window:
                    break
                begin = random.randint(begin, dur//2-len_window)
                new_len = random.randint(1, len_window)
                end = begin + new_len
                if wrong_flag[k] == 0:
                    new_clips = [idx for idx in range(begin, end)]
                    new_window = [begin*2, end*2]
                    new_clip_ids += new_clips
                    new_relevant_windows.append(new_window)
                
                aug_frames_clip_feat = aug_clip_feat[aug_windows[k][0]//2:aug_windows[k][1]//2]
                aug_frames_sf_feat = aug_sf_feat[aug_windows[k][0]//2:aug_windows[k][1]//2]
                assert len(aug_frames_clip_feat) == len(aug_frames_sf_feat)
                assert len(aug_frames_clip_feat) == len_window
                # random sample new_len frames from aug_frames_feat
                sample_idx = random.sample(range(len_window), new_len)
                sample_idx.sort()
                aug_frames_clip_feat = aug_frames_clip_feat[sample_idx]
                aug_frames_sf_feat = aug_frames_sf_feat[sample_idx]
                new_clip_feat[begin:end] = aug_frames_clip_feat
                new_sf_feat[begin:end] = aug_frames_sf_feat
                if wrong_flag[k] == 0:
                    new_sampled_scores = [new_scores[idx] for idx in sample_idx]
                    new_saliency_scores += new_sampled_scores
                begin = end + 1
            new_data['relevant_clip_ids'] = new_clip_ids
            new_data['saliency_scores'] = new_saliency_scores
            new_data['relevant_windows'] = new_relevant_windows
            np.save(f'{aug_clip_feat_save_path}/{new_data["vid"]}.npy', new_clip_feat)
            np.savez_compressed(f'{aug_sf_feat_save_path}/{new_data["vid"]}.npz', features=new_sf_feat)
            augmented_data.append(new_data)
        except Exception as e:
            continue

i = 0
while i < len(augmented_data):
    d = augmented_data[i]
    clip_ids = d["relevant_clip_ids"]
    if len(clip_ids) == 0:
        # remove the empty clip_ids
        augmented_data.pop(i)
    else:
        i += 1

with open(save_path, 'w') as f:
    for line in augmented_data:
        json.dump(line, f)
        f.write('\n')
print(f'Augmented data saved to {save_path}')