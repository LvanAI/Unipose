# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""callback function"""
import os
import math
import time
from copy import deepcopy


import mindspore.ops as ops
from mindspore import nn

from mindspore.train.callback import Callback
from mindspore import save_checkpoint
from collections import namedtuple
from mindspore import Tensor, float32
import numpy as np

from src.args import args

def calc_dists(preds, target, normalize):
	preds  =  preds.astype(np.float32)
	target = target.astype(np.float32)
	dists  = np.zeros((preds.shape[1], preds.shape[0]))

	for n in range(preds.shape[0]):
		for c in range(preds.shape[1]):
			if target[n, c, 0] > 1 and target[n, c, 1] > 1:
				normed_preds   =  preds[n, c, :] / normalize[n]
				normed_targets = target[n, c, :] / normalize[n]
				dists[c, n]    = np.linalg.norm(normed_preds - normed_targets)
			else:
				dists[c, n]    = -1

	return dists


def dist_acc(dists, threshold = 0.5):
	dist_cal     = np.not_equal(dists, -1)
	num_dist_cal = dist_cal.sum()

	if num_dist_cal > 0:
		return np.less(dists[dist_cal], threshold).sum() * 1.0 / num_dist_cal
	else:
		return -1


def get_max_preds(batch_heatmaps):
	batch_size = batch_heatmaps.shape[0]
	num_joints = batch_heatmaps.shape[1]
	width      = batch_heatmaps.shape[3]

	heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
	idx               = np.argmax(heatmaps_reshaped.asnumpy(), 2)
	maxvals           = np.amax(heatmaps_reshaped.asnumpy(), 2)

	maxvals = maxvals.reshape((batch_size, num_joints, 1))
	idx     = idx.reshape((batch_size, num_joints, 1))

	preds   = np.tile(idx, (1,1,2)).astype(np.float32)

	preds[:,:,0] = (preds[:,:,0]) % width
	preds[:,:,1] = np.floor((preds[:,:,1]) / width)

	pred_mask    = np.tile(np.greater(maxvals, 0.0), (1,1,2))
	pred_mask    = pred_mask.astype(np.float32)

	preds *= pred_mask

	return preds, maxvals


def accuracy(output, target, thr_PCK, thr_PCKh, hm_type='gaussian', threshold=0.5):
	idx  = list(range(output.shape[1]))
	norm = 1.0

	if hm_type == 'gaussian':
		pred, _   = get_max_preds(output)
		target, _ = get_max_preds(target)

		h         = output.shape[2]
		w         = output.shape[3]
		norm      = np.ones((pred.shape[0], 2)) * np.array([h,w]) / 10

	dists = calc_dists(pred, target, norm)

	acc     = np.zeros((len(idx)))
	avg_acc = 0
	cnt     = 0
	visible = np.zeros((len(idx)))

	for i in range(len(idx)):
		acc[i] = dist_acc(dists[idx[i]])
		if acc[i] >= 0:
			avg_acc = avg_acc + acc[i]
			cnt    += 1
			visible[i] = 1
		else:
			acc[i] = 0

	avg_acc = avg_acc / cnt if cnt != 0 else 0

	if cnt != 0:
		acc[0] = avg_acc

	# PCKh
	PCKh = np.zeros((len(idx)))
	avg_PCKh = 0


	headLength = np.linalg.norm(target[0,14,:] - target[0,13,:])


	for i in range(len(idx)):
		PCKh[i] = dist_acc(dists[idx[i]], thr_PCKh*headLength)
		if PCKh[i] >= 0:
			avg_PCKh = avg_PCKh + PCKh[i]
		else:
			PCKh[i] = 0

	avg_PCKh = avg_PCKh / cnt if cnt != 0 else 0

	if cnt != 0:
		PCKh[0] = avg_PCKh


	# PCK
	PCK = np.zeros((len(idx)))
	avg_PCK = 0

	pelvis = [(target[0,3,0]+target[0,4,0])/2, (target[0,3,1]+target[0,4,1])/2]
	torso  = np.linalg.norm(target[0,13,:] - pelvis)

	for i in range(len(idx)):
		PCK[i] = dist_acc(dists[idx[i]], thr_PCK*torso)

		if PCK[i] >= 0:
			avg_PCK = avg_PCK + PCK[i]
		else:
			PCK[i] = 0

	avg_PCK = avg_PCK / cnt if cnt != 0 else 0

	if cnt != 0:
		PCK[0] = avg_PCK


	return acc, PCK, PCKh, cnt, pred, visible


def printAccuracies(mPCKh, PCKh, mPCK, PCK):
	print("mPCK@0.2:  %.2f%%" % (mPCK * 100))
	print("mPCKh@0.5: %.2f%%" % (mPCKh * 100))


def validation(model, val_dataset,loss_fn, cur_epoch_num, cur_lr, src_url, dst_url):
	model.set_train(False)
	val_loss = 0.0
	# 定义内存空间，每幅图像计算的PCK、PCKh共用一个内存
	PCK = np.zeros(args.num_classes + 1)
	PCKh = np.zeros(args.num_classes + 1)
	count = np.zeros(args.num_classes + 1) # count存放已计算的样本数
	mPCK = 0
	mPCKh = 0

	i,cnt = 0,0

	for item in val_dataset.create_dict_iterator(num_epochs=1):
		inputs = item['img']
		outputs = model(inputs)
		heatmap = item['heatmap']
		loss = loss_fn(outputs,heatmap)
		val_loss += loss

		acc, acc_PCK, acc_PCKh, cnt, pred, visible = accuracy(outputs,heatmap,0.2, 0.5)

		PCK[0] = (PCK[0] * i + acc_PCK[0]) / (i + 1)
		PCKh[0] = (PCKh[0] * i + acc_PCKh[0]) / (i + 1)

		for j in range(1, args.num_classes + 1):
			if visible[j] == 1:
				PCK[j] = (PCK[j] * count[j] + acc_PCK[j]) / (count[j] + 1)
				PCKh[j] = (PCKh[j] * count[j] + acc_PCKh[j]) / (count[j] + 1)
				count[j] += 1

		mPCK = PCK[1:].sum() / (args.num_classes)
		mPCKh = PCKh[1:].sum() / (args.num_classes)

		i+=1

	print('Val   loss: ' ,(val_loss / (i * args.batch_size)))
	# printAccuracies(mPCKh, PCKh, mPCK, PCK)

	PCKhAvg = PCKh.sum() / (args.num_classes + 1)
	PCKAvg = PCK.sum() / (args.num_classes + 1)


	if mPCK > args.bestPCK:
		args.best_epoch = cur_epoch_num
		args.bestPCK = mPCK
		save_checkpoint(model, os.path.join(src_url, 'unipose_best.ckpt'))

		if args.run_modelarts:
			import moxing as mox
			mox.file.copy_parallel(src_url = src_url, dst_url = dst_url)


	if mPCKh > args.bestPCKh:
		args.bestPCKh = mPCKh

	print("Current epoch %d ,   PCK@0.2 = %2.2f%%, PCKh@0.5 = %2.2f%%" % (
		 cur_epoch_num,  mPCK * 100, mPCKh * 100))
	print("Best epoch %d , PCK@0.2 = %2.2f%%, PCKh@0.5 = %2.2f%%" % (
		 args.best_epoch,args.bestPCK * 100, args.bestPCKh * 100))


class EvaluateCallBack(Callback):
	"""
	EvaluateCallBack
	"""
	def __init__(self, model, eval_dataset, loss_fn, lr, src_url, dst_url):
		super(EvaluateCallBack, self).__init__()
		self.dst_url = dst_url
		self.src_url = src_url
		self.lr = lr
		self.model = model
		self.eval_dataset = eval_dataset
		self.loss = loss_fn
		

	def epoch_end(self, run_context):
		"""
		Test when epoch end, save best model with best.ckpt.
		"""
		cb_params = run_context.original_args()
		cur_epoch_num = cb_params.cur_epoch_num
		validation(self.model, val_dataset=self.eval_dataset, loss_fn=self.loss, cur_epoch_num=cur_epoch_num, cur_lr=self.lr[cb_params.cur_step_num-1], src_url = self.src_url, dst_url = self.dst_url)



