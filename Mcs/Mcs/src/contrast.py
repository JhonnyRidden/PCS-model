import torch
import torch.nn as nn
import random


class CLBlock(nn.Module):
    """
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy
        reduction: Reduction method applied to the output. Values must be one of ['none', 'sum', 'mean'].
        negative_mode: Determines how the (optional) negative_keys are handled. Values must be one of ['unpaired', 'paired'].
            If paired, then each query sample is paired with a number of negative keys. Comparable to a triplet loss,
            but with multiple negatives per sample.
            If unpaired, then the set of negative keys are all unrelated to any positive keys.

    Input Shape:
        query: (N, D) Tensor with query items (e.g. embedding of the input)
        key_matrix: (N, N) Tensor to indicate the positive and negative samples of target query items.

    Returns:
        Value of the InfoNCE Loss.
    """

    def __init__(self, config):
        super().__init__()
        self.temperature = 0.2
        self.base_temperature = 1
        self.device = config['device']
        self.contrast_mode = 'one'
        # self.positive_sample_encoder = nn.MultiheadAttention(self.dim, self.num_heads)

    def forward(self, features):
        labels = self.requires_grad_(features)
        cl_loss = self.contrast(features, labels)
        return cl_loss

    def setup_contrast_label(self, features):
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        similarity_matrix = torch.clamp(similarity_matrix, min=0, max=1)
        labels = torch.round(similarity_matrix)
        return labels

    def contrast(self, features, labels):
        bsz = features.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        mask = torch.ones_like(similarity_matrix) * (labels.expand(bsz, bsz).eq(labels.expand(bsz, bsz).t()))
        mask_no_sim = torch.ones_like(mask) - mask
        mask_eye_0 = (torch.ones(bsz, bsz) - torch.eye(bsz, bsz)).to(self.device)
        similarity_matrix = torch.exp(similarity_matrix / self.temperature)
        similarity_matrix = similarity_matrix * mask_eye_0
        sim = mask*similarity_matrix
        no_sim = similarity_matrix - sim
        no_sim_sum = torch.sum(no_sim, dim=1)
        no_sim_sum_expand = no_sim_sum.repeat(bsz, 1).T
        sim_sum = sim + no_sim_sum_expand
        loss = torch.div(sim, sim_sum)
        loss = mask_no_sim + loss + torch.eye(bsz, bsz).to(self.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1))/(2*bsz)
        return loss

    def simple_contrast(self, features, labels, mask):
        if len(features.shape) < 3:
            features = features.unsqueeze(-1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and masks')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels dose not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        elif self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        else:
            raise ValueError('Unknown mode')

        # Compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def building_cl_samples(self, query, key_matrix, fea_masker, fea_shared, fea_visual, fea_audio, fea_text):
        # Check if the dimensions are correct.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if (key_matrix.dim() != 2) | (key_matrix.shape[0] != key_matrix.shape[1]):
            raise ValueError('<key_matrix> must have 2 dimensions, and the two dimensions should be the same.')

        # Constructing negative samples
        M = self.negative_sample_num
        N = query.shape[0]
        if self.negative_mode == 'paired':
            # If 'paired', then each query item is paired with a certain number of negative keys
            # The output negative_keys is a (N, M, D) Tensor.
            negative_sampled_list1 = []
            for i in range(N):
                keys_slice = key_matrix[:, i]
                negative_total = torch.sum(keys_slice == 0)
                assert negative_total >= M
                # Make sure the negative sample number is smaller than negative total number

                zeros_indices = torch.nonzero(keys_slice == 0)[:, 0]
                indices = random.sample(zeros_indices.tolist(), k=M)

                sampled_slice_list2 = []
                for idx in indices:
                    if idx == 0:
                        sampled_slice_list2.append(query[:1, :])
                    else:
                        sampled_slice_list2.append(query[idx:idx+1, :])
                negative_samples = torch.cat(sampled_slice_list2, dim=0).unsqueeze(dim=0)
                negative_sampled_list1.append(negative_samples)
            negative_samples = torch.cat(negative_sampled_list1, dim=0)
        elif self.negative_mode == 'unpaired':
            sum_vector = torch.sum(key_matrix, dim=1, keepdim=True)
            indices = torch.nonzero(sum_vector == 1)[:, 0]
            sampled_indices = random.sample(indices.tolist(), k=M)

            sampled_tensor_list = []
            for idx in sampled_indices:
                sampled_tensor_list.append(query[idx:idx+1, :])
            negative_samples = torch.cat(sampled_tensor_list, dim=0)
        else:
            raise ValueError('Unsupported negative sampling modes')

        # Constructing positive samples
        sum_vector = torch.sum(key_matrix, dim=0, keepdim=True).squeeze()
        no_matching_list = torch.nonzero(sum_vector == 1)[:, 0].tolist()
        matching_list = torch.nonzero(sum_vector != 1)[:, 0].tolist()

        positive_samples_dict = {}
        for i in range(N):
            if i in no_matching_list:
                masker = fea_masker[i, :]
                masker = [masker[1].int(), masker[2].int(), masker[3].int()]
                masker_indices = [i for i, value in enumerate(masker) if value == 0]
                positive_sample_list = []
                if 0 in masker_indices:
                    positive_sample_list.append(fea_visual[i, :].unsqueeze(dim=0))
                if 1 in masker_indices:
                    positive_sample_list.append(fea_audio[i, :].unsqueeze(dim=0))
                if 2 in masker_indices:
                    positive_sample_list.append(fea_text[i, :].unsqueeze(dim=0))
                positive_samples = torch.cat(positive_sample_list, dim=0)
                positive_samples_dict.update({i: positive_samples})
            elif i in matching_list:
                key_slice = key_matrix[:, i]
                assert torch.sum(key_slice) > 1
                key_indices = [i for i, value in enumerate(key_slice) if value == 1]
                fea_shared_selected = fea_shared[key_indices, :]
                positive_samples_dict.update({i: fea_shared_selected})
            else:
                raise ValueError('The inner process reports error, please check the code')

        return positive_samples_dict, negative_samples