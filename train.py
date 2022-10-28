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
"""train"""
import imp
import os

from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore import ops
from mindspore import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import JointsMSELoss, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, pretrained
from src.tools.optimizer import get_optimizer


def main():
    assert args.crop, f"{args.arch} is only for evaluation"
    set_seed(args.seed)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[args.graph_mode], device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    if args.device_target == "Ascend":
        context.set_context(enable_auto_mixed_precision=True)
        
    rank = set_device(args)

    # get model and cast amp_level
    if args.pretrained:
        import sys
        args.pretrained = sys.path[0] + '/' + args.pretrained
    net = get_model(args)
    cast_amp(net)

    criterion = JointsMSELoss()
    net_with_loss = NetWithLoss(net, criterion)

    # if args.pretrained:
    #     pretrained(args, net)

    # get data
    data = get_dataset(args)
    batch_num = data.train_dataset.get_dataset_size()
    print('batch_num %d'% (batch_num))
    optimizer, learning_rate= get_optimizer(args, net, batch_num)
    # save a yaml file to read to record parameters

    model = Model(network=net_with_loss, optimizer = optimizer, amp_level= args.amp_level)

    config_ck = CheckpointConfig(save_checkpoint_steps=data.train_dataset.get_dataset_size(),
                                 keep_checkpoint_max=args.save_every)

    time_cb = TimeMonitor(data_size = data.train_dataset.get_dataset_size())

    ckpt_save_dir = "./checkpoint/ckpt_" + str(rank)
    dst_url=os.path.join(args.train_url, "ckpt_" + str(rank))

    if args.run_modelarts:
        ckpt_save_dir = "/cache/ckpt_" + str(rank)

    os.makedirs(ckpt_save_dir, exist_ok=True)
    #ckpoint_cb = ModelCheckpoint(prefix = args.arch + str(rank), directory = ckpt_save_dir,
    #                             config=config_ck)
    
    callbacks=[time_cb]

    eval_cb = EvaluateCallBack(net, eval_dataset=data.val_dataset, loss_fn = criterion, lr = learning_rate, src_url = ckpt_save_dir, dst_url = dst_url)
                           
    callbacks += [eval_cb]
    
    print("begin train")
    model.train(int(args.epochs - args.start_epoch), data.train_dataset,
                callbacks= callbacks,
                dataset_sink_mode = False)
                
    print("train success")



if __name__ == '__main__':
    main()
