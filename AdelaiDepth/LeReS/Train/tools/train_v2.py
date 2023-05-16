import math
import os, sys
import errno

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from data.load_dataset_distributed import MultipleDataLoaderDistributed, MultipleDatasetDistributed
from lib.models.multi_depth_model_auxiv2 import *
from lib.configs.config import cfg, merge_cfg_from_file, print_configs
from lib.utils.net_tools import save_ckpt, load_ckpt
from tools.parse_arg_base import print_options
from lib.configs.config import cfg, merge_cfg_from_file, print_configs

from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_rel_depth_err, recover_metric_depth
from lib.utils.lr_scheduler_custom import make_lr_scheduler
from lib.utils.logging import setup_distributed_logger, SmoothedValue


## for dataloaders
import torch.utils.data
import argparse
import copy

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--logdir", default="log_test/", help="path to the log directory", type=str)

### Load pretrained model
parser.add_argument("--ckpt", default="/orion/u/mikacuy/coordinate_mvs/AdelaiDepth/LeReS/Minist_Test/res101.pth", help="checkpoint", type=str)

parser.add_argument('--loss_mode', type=str, default='_ranking-edge_pairwise-normal-regress-edge_msgil-normal_meanstd-tanh_pairwise-normal-regress-plane_', help='losses to use')
parser.add_argument('--epoch', default= 600, type=int)
parser.add_argument('--batchsize', default= 4, type=int)

parser.add_argument('--base_lr', default= 0.001, type=float)
parser.add_argument('--lr_scheduler_multiepochs', default=[10, 25, 40], nargs='+', type=int, help='Learning rate scheduler step')

parser.add_argument('--thread', default= 4, type=int)
parser.add_argument('--use_tfboard', default= True, type=bool)

### For the dataset
parser.add_argument('--phase', type=str, default='train', help='Training flag')
parser.add_argument('--phase_anno', type=str, default='train', help='Annotations file name')
parser.add_argument('--dataset_list', default=["taskonomy"], nargs='+', help='The names of multiple datasets')
parser.add_argument('--dataset', default='multi', help='Dataset loader name')
parser.add_argument('--dataroot', default='/orion/downloads/coordinate_mvs/', help='Root dir for dataset')


parser.add_argument('--backbone', default= "resnext101", type=str)
parser.add_argument('--d_latent', default= 32, type=int)
parser.add_argument('--num_samples', default= 20, type=int, help='Number of z codes to sample')
parser.add_argument('--refresh_z', default= 10, type=int, help='Number of epochs of when to recache z')

FLAGS = parser.parse_args()
LOG_DIR = FLAGS.logdir
CKPT = FLAGS.ckpt

D_LATENT = FLAGS.d_latent
NUM_SAMPLE = FLAGS.num_samples

MAX_EPOCH = FLAGS.epoch
REFRESH_Z = FLAGS.refresh_z

##############################################
###### Dataset utils for cIMLE implementation
##############################################
class ZippedDataset(torch.utils.data.Dataset):

    def __init__(self, *datasets):
        assert all(len(datasets[0]) == len(dataset) for dataset in datasets)
        self.datasets = datasets

    def __getitem__(self, index):
        return tuple(dataset[index] for dataset in self.datasets)

    def __len__(self):
        return len(self.datasets[0])

class ChoppedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, num_elems):
        self.dataset = dataset
        self.num_elems = num_elems

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(len(self.dataset), self.num_elems)


class SlicedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, slice_indices):
        self.dataset = dataset
        self.slices = slice_indices

    def __getitem__(self, index):
        return tuple(self.dataset[index][s] for s in self.slices)

    def __len__(self):
        return len(self.dataset)
##############################################


### Merge config with current configs
merge_cfg_from_file(FLAGS)

##### Set-up and all... ####
cfg.TRAIN.RUN_NAME = FLAGS.logdir.rstrip("/")
cfg.TRAIN.OUTPUT_DIR = './'
cfg.TRAIN.LOG_DIR = os.path.join(cfg.TRAIN.OUTPUT_DIR, cfg.TRAIN.RUN_NAME)

log_output_dir = cfg.TRAIN.LOG_DIR
if log_output_dir:
    try:
        os.makedirs(log_output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

### Disabled distributed training
local_rank = 0
world_size = 1

logger = setup_distributed_logger("lib", log_output_dir, local_rank, cfg.TRAIN.RUN_NAME + '.txt')
tblogger = None
if FLAGS.use_tfboard and  local_rank == 0:
    from tensorboardX import SummaryWriter
    tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

val_err = [{'abs_rel': 0, 'whdr': 0}]
############################

model = RelDepthModel_cIMLE(d_latent=D_LATENT)
model.cuda()

### Load model
model_dict = model.state_dict()
CKPT_FILE = CKPT

if os.path.isfile(CKPT_FILE):
    print("loading checkpoint %s" % CKPT_FILE)
    checkpoint = torch.load(CKPT_FILE)

    ### Need to check if data parallel
    checkpoint['depth_model'] = strip_prefix_if_present(checkpoint['depth_model'], "module.")
    depth_keys = {k: v for k, v in checkpoint['depth_model'].items() if k in model_dict} ## <--- some missing keys in the loaded model from the given model
    print(len(depth_keys))

    if (len(depth_keys) == 0):
    	print("Error in loading pretrained model.")
    	exit()

    # Overwrite entries in the existing state dict
    model_dict.update(depth_keys)        

    # Load the new state dict
    model.load_state_dict(model_dict)
    print("Model loaded.")

### Dataset
train_dataset = MultipleDatasetDistributed(FLAGS)

## Get val dataset
val_args = copy.deepcopy(FLAGS)
val_args.phase_anno = "val"
val_args.phase = "val"
val_dataset = MultipleDatasetDistributed(val_args)

print("Datasets:")
print("Train")
print(len(train_dataset))
print()
print("Val:")    
print(len(val_dataset))
print("====================")

### Set up learning rate and optimizer
cfg.TRAIN.LR_SCHEDULER_MULTISTEPS = np.array(FLAGS.lr_scheduler_multiepochs) * math.ceil(len(train_dataset)/ (world_size * FLAGS.batchsize))
optimizer = ModelOptimizer(model)
scheduler = make_lr_scheduler(cfg=cfg, optimizer=optimizer.optimizer)
total_iters = math.ceil(len(train_dataset)/ (world_size * FLAGS.batchsize)) * FLAGS.epoch
cfg.TRAIN.MAX_ITER = total_iters
cfg.TRAIN.GPU_NUM = world_size
print_configs(cfg)

training_stats = TrainingStats(FLAGS, cfg.TRAIN.LOG_INTERVAL, tblogger if FLAGS.use_tfboard else None)

### Dataloader unshuffled to cache z-codes
zcache_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    num_workers=FLAGS.thread,
    shuffle=False)

print(len(zcache_dataloader))
print()

### Minibatch to handle larger sample size
mini_batch_size = 10
num_sets = int(NUM_SAMPLE/mini_batch_size)
true_num_samples = num_sets*mini_batch_size # just take the floor

model.train()
tmp_i = 0

for epoch in range(MAX_EPOCH):

    if epoch == 0 or epoch%REFRESH_Z  == 0:
        ### Resample z and take the best one

        ### Set network to eval mode
        model.eval()

        ### Iterate over dataset
        selected_z_np = np.empty((len(zcache_dataloader),D_LATENT), dtype=np.float32)
        print("Size of latent code matrix")
        print(selected_z_np.shape)
        print()

        
        with torch.no_grad():
            for i, data in enumerate(zcache_dataloader):

                batch_size = data['rgb'].shape[0]
                C = data['rgb'].shape[1]
                H = data['rgb'].shape[2]
                W = data['rgb'].shape[3]

                ### Loss values
                all_losses = torch.zeros((batch_size, true_num_samples)).cuda()
                all_z = torch.zeros((batch_size, true_num_samples, D_LATENT)).cuda()

                ### Repeat for the number of samples
                num_images = data['rgb'].shape[0]
                data['rgb'] = data['rgb'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
                data['rgb'] = data['rgb'].view(-1, C, H, W)
                data['depth'] = data['depth'].unsqueeze(1).repeat(1,mini_batch_size, 1, 1, 1)
                data['depth'] = data['depth'].view(-1, 1, H, W)            
                data['quality_flg'] = data['quality_flg'].unsqueeze(1).repeat(1,mini_batch_size)
                data['quality_flg'] = data['quality_flg'].view(-1)
                data['focal_length'] = data['focal_length'].unsqueeze(1).repeat(1,mini_batch_size)
                data['focal_length'] = data['focal_length'].view(-1)
                data['planes'] = data['planes'].unsqueeze(1).repeat(1,mini_batch_size,1,1)
                data['planes'] = data['planes'].view(-1, H, W)

                ### Iterate over the minibatch
                for k in range(num_sets):

                    ## Hard coded d_latent
                    z = torch.normal(0.0, 1.0, size=(num_images, mini_batch_size, D_LATENT))
                    z = z.view(-1, D_LATENT).cuda()

                    out = model(data, z)
                    losses_dict, total_raw = out['losses']
                    total_raw = total_raw.view(batch_size, mini_batch_size)
                    z = z.view(batch_size, mini_batch_size, D_LATENT)

                    for s in range(mini_batch_size):
                        all_losses[:, k*mini_batch_size + s] = total_raw[:, s]
                        all_z[:, k*mini_batch_size + s, :] = z[:, s, :]


                all_z = all_z.view(batch_size, true_num_samples, D_LATENT)

                idx_to_take = torch.argmin(all_losses, axis=-1)

                for j in range(batch_size):
                    selected_z_np[i*batch_size+j,:] = all_z[j][idx_to_take[j]].cpu().data.numpy()

                if i%100==0:
                    print("Caching "+str(i)+"/"+str(len(zcache_dataloader))+".")

                torch.cuda.empty_cache()

        ### Save to file
        cur_z_data_file = "z_data_epoch_%d.npy" % (epoch)
        np.save(os.path.join(LOG_DIR, cur_z_data_file), selected_z_np)

        ### Reset to train network
        model.train()

        print()
        print("Finished caching z-codes...")
        print(selected_z_np.shape)

    ### Create dataset with selected z
    print("Creating combined dataloader")
    comb_dataset = ZippedDataset(train_dataset, torch.utils.data.TensorDataset(torch.from_numpy(selected_z_np)))
    train_dataloader = torch.utils.data.DataLoader(
        dataset=comb_dataset,
        batch_size=FLAGS.batchsize,
        num_workers=FLAGS.thread-1,
        shuffle=True, pin_memory=True)

    print("Start training")
    ### Iterate over shuffled dataset and train
    for i, (data, (cur_batch_z,)) in enumerate(train_dataloader):

        cur_batch_z = cur_batch_z.cuda()
        out = model(data, cur_batch_z)
        losses_dict, total_raw = out['losses']
        optimizer.optim(losses_dict)
        tmp_i += 1

        # reduce losses over all GPUs for logging purposes  --> unimplemented
        # loss_dict_reduced = reduce_loss_dict(losses_dict)
        loss_dict_reduced = losses_dict

        scheduler.step()
        training_stats.UpdateIterStats(loss_dict_reduced)
        training_stats.IterToc()
        training_stats.LogIterStats(tmp_i, epoch, optimizer.optimizer)


    print("Epoch "+str(epoch)+"/"+str(MAX_EPOCH)+".")
    
    # save checkpoint
    if epoch % 8 == 0:
        print("Saving to model...")
        save_ckpt(FLAGS, tmp_i, epoch, model, optimizer.optimizer, scheduler, total_raw)



print("Saving to model...")
save_ckpt(FLAGS, tmp_i, epoch, model, optimizer.optimizer, scheduler, total_raw)

print("Done.")


