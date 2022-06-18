from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.optim import AdamW, SGD
from tqdm import tqdm

from dataset.dg_dataset import *
from network.components.customized_evaluate import NaturalImageMeasure, MeterDicts
from network.components.schedulers import PolyLR

from resnet import Net
from segformer_model import segformer
from segformer_lawin_model import segformer_lawin
# from torch.nn.parallel import DistributedDataParallel

from utils.nn_utils import *
from utils.nn_utils import get_updated_network, get_logger, get_img_target
from utils.visualize import show_graphs

#from utils.utils.loss import CrossEntropy2d
#from utils.utils.loss import CrossEntropyLoss2dPixelWiseWeighted

import torch.backends.cudnn as cudnn

from PIL import Image
from torchvision import transforms
import torchvision
# from memory_profiler import profile

# 123456
seed = 123456
torch.manual_seed(seed)
np.random.seed(seed)


class MetaFrameWork(object):
    def __init__(self, name='normal_all', train_num=1, source='CI', opt='A',
                 target='D', network='O', resume=True, dataset=DGMetaDataSets,
                 inner_lr=5e-4, outer_lr=2e-3, train_size=8, test_size=16, no_source_test=True, bn='torch', mix = 2):
        super(MetaFrameWork, self).__init__()
        self.no_source_test = no_source_test
        self.train_num = train_num
        self.exp_name = name
        self.resume = resume
        self.mix = 2

        self.inner_update_lr = inner_lr
        self.outer_update_lr = outer_lr
        
        self.network = network
        self.opt = opt
            
        self.dataset = dataset
        self.train_size = train_size
        self.test_size = test_size
        self.source = source
        self.target = target
        self.bn = bn

        self.epoch = 1
        self.best_target_acc = 0
        self.best_target_acc_source = 0
        self.best_target_epoch = 1

        self.best_source_acc = 0
        self.best_source_acc_target = 0
        self.best_source_epoch = 0

        self.total_epoch = 120
        self.accumulation_steps = 1 
        self.save_interval = 1
        self.save_path = Path(self.exp_name)
        self.init()

    def init(self):
        
        # cudnn.enabled = True
        
        kwargs = {'bn': self.bn, 'output_stride': 8}
        
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
        # PixelWiseWeighted LOSS
        # self.unlabeled_loss = nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=-1)).cuda()
        
        if self.network =='O':
            self.backbone = segformer_lawin(backbone='mit_b2', num_classes=19, embedding_dim=768, pretrained=True).cuda()
            param_groups = self.backbone.get_param_groups()
            self.backbone = nn.DataParallel(self.backbone)
            self.updated_net = nn.DataParallel(segformer_lawin(backbone='mit_b2', num_classes=19, embedding_dim=768, pretrained=True).cuda())
            
            
        elif self.network == 'S':
            self.backbone = segformer(backbone='mit_b2', num_classes=19, embedding_dim=768, pretrained=True).cuda()
            param_groups = self.backbone.get_param_groups()
            self.backbone = nn.DataParallel(self.backbone)
            self.updated_net = nn.DataParallel(segformer(backbone='mit_b2', num_classes=19, embedding_dim=768, pretrained=True).cuda())
                
        elif self.network == 'R':
            self.backbone = nn.DataParallel(Net(**kwargs)).cuda()
            kwargs.update({'pretrained': False})
            self.updated_net = nn.DataParallel(Net(**kwargs)).cuda()  
        
        self.nim = NaturalImageMeasure(nclass=19)

        batch_size = self.train_size
        workers = len(self.source) 
        
        # source
        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=batch_size, shuffle=True)
        self.train_loader = dataloader(self.dataset(mode='train', domains=self.source, force_cache=False, crop_size=512, imgs_per_epoch=3000))

        # dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=self.test_size, shuffle=False)
        # self.source_val_loader = dataloader(self.dataset(mode='val', domains=self.source, force_cache=True))
        
        # traget
        target_dataset, folder = get_dataset(self.target)
        self.target_loader_train = dataloader(target_dataset(root=ROOT + folder, mode='test'), batch_size=batch_size, crop_size=512)
        #self.target_loader_train = dataloader(target_dataset(root=ROOT + folder, mode='test'), batch_size=len(self.source)*batch_size)

        dataloader = functools.partial(DataLoader, num_workers=workers, pin_memory=True, batch_size=self.test_size, shuffle=False)
        self.target_loader_val = dataloader(target_dataset(root=ROOT + folder, mode='val'))       
        # self.target_test_loader = dataloader(target_dataset(root=ROOT + folder, mode='test'))
        if self.opt == 'S':
            self.opt_old = SGD(self.backbone.parameters(), lr=self.outer_update_lr, momentum=0.9, weight_decay=5e-4)
        elif self.opt == 'A':
            self.opt_old = AdamW(params=[
                    {
                        "params": param_groups[0],
                        "lr": self.outer_update_lr,
                        "weight_decay": 5e-4,
                    },
                    {
                        "params": param_groups[1],
                        "lr": self.outer_update_lr,
                        "weight_decay": 0.0,
                    },
                    {
                        "params": param_groups[2],
                        "lr": self.outer_update_lr * 10,
                        "weight_decay": 5e-4,
                    },
                ], lr=self.outer_update_lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=5e-4)
            
        self.scheduler_old = PolyLR(self.opt_old, self.total_epoch, len(self.train_loader), 0, True, power=0.9)
        
        self.interp = nn.Upsample(size=(768, 768), mode='bilinear', align_corners=True)

        self.logger = get_logger('train', self.exp_name)
        self.log('exp_name : {}, train_num = {}, mix = {}, source domains = {}, target_domain = {}, lr : inner = {}, outer = {},'
                 'dataset : {}, net : {}, bn : {}\n'.
                 format(self.exp_name, self.train_num, self.mix, self.source, self.target, self.inner_update_lr, self.outer_update_lr, self.dataset,
                        self.network, self.bn))
        self.log(self.exp_name + '\n')
        self.train_timer, self.test_timer = Timer(), Timer()

    def train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        imgs, targets = inputs
        N, C, H, W = imgs.size()
        # mixed only for meta test
        
        meta_train_imgs = imgs
        meta_train_targets = targets

        tr_logits = self.backbone(meta_train_imgs)
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])
        with torch.no_grad():
            self.nim(tr_logits, meta_train_targets)
            
        
        # self.opt_old.zero_grad(set_to_none=True)
        ds_loss.backward()
        
        self.opt_old.step()
        for param in self.backbone.parameters():
            param.grad=None
            
        self.scheduler_old.step(epoch, it)
        losses = {
            'dg': 0,
            'ds': ds_loss.item()
        }
        acc = {
            'iou': self.nim.get_res()[0]
        }
        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]

    # @profile
    def meta_train(self, epoch, it, inputs):
        # imgs : batch x domains x C x H x W
        # targets : batch x domains x 1 x H x W
        
        # use source images for meta-train, mixed images for meta-test
        imgs, targets = inputs
        N, C, H, W = imgs.size()
        # mixed only for meta test
        if self.mix == 1:
            train_idx = split_idx[:N // 2]
            test_idx = split_idx[N // 2:]
        else:
            split_idx = np.random.permutation(N)
            i = np.random.randint(1, N)
            train_idx = split_idx[:i]
            test_idx = split_idx[i:]
        
        meta_train_imgs = imgs[train_idx]
        meta_train_targets = targets[train_idx]
        meta_test_imgs = imgs[test_idx]
        meta_test_targets = targets[test_idx]
        
        # Meta-Train
        tr_logits = self.backbone(meta_train_imgs)
        tr_logits = make_same_size(tr_logits, meta_train_targets)
        ds_loss = self.ce(tr_logits, meta_train_targets[:, 0])

        # Update new network
        # for param in self.backbone.parameters():
                # param.grad=None
        self.opt_old.zero_grad(set_to_none=True)
  
        ds_loss.backward(retain_graph=True)
        
        updated_net = get_updated_network(self.backbone, self.updated_net, self.inner_update_lr).train().cuda()

        # Meta-Test
        te_logits = updated_net(meta_test_imgs)
        # te_logits = test_res[0]
        te_logits = make_same_size(te_logits, meta_test_imgs)    
            

        dg_loss = self.ce(te_logits, meta_test_targets[:, 0])
        
        with torch.no_grad():
            self.nim(te_logits, meta_test_targets)

        # Update old network
        dg_loss.backward()
        self.opt_old.step()
        
        self.scheduler_old.step(epoch, it)
        losses = {
            'dg': dg_loss.item(),
            'ds': ds_loss.item()
        }
        acc = {
            'iou': self.nim.get_res()[0],
        }
        return losses, acc, self.scheduler_old.get_lr(epoch, it)[0]
        
    # @profile
    def do_train(self):
        if self.resume:
            # self.load('best_dada_seg')
            self.load() 

        self.writer = SummaryWriter(str(self.save_path / 'tensorboard'), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S'))
        self.log('Start epoch : {}\n'.format(self.epoch))

        for epoch in range(self.epoch, self.total_epoch + 1):
            loss_meters, acc_meters = MeterDicts(), MeterDicts(averaged=['iou'])
            self.nim.clear_cache()
            self.backbone.train()
            self.updated_net.train()
            
            self.epoch = epoch
            with self.train_timer:
                for it, (s, t) in enumerate(zip(self.train_loader, self.target_loader_train)):
                    
                    meta = (it + 1) % self.train_num == 0
                    mix = True
                    s_paths, s_imgs, s_targets = s
                    t_paths, t_imgs = t
                    # source
                    s_imgs, s_targets =  to_cuda([s_imgs, s_targets])
                    B, D, C, H, W = s_imgs.size()
                    #print(B, D, C, H, W)
                    s_imgs = s_imgs.view(-1, C, H, W)
                    s_targets = s_targets.view(-1, 1, H, W)
                    
                    if self.mix != 0: 
                    
                        # target train images, labels  
                         
                        t_imgs =  to_cuda(t_imgs)
                        #print(t_imgs.size())
                        B, C, H, W = t_imgs.size()
                        
                        # pseudo_label: target for training
                        logits_u_w = self.backbone(t_imgs)
                        #logits_u_w = nn.functional.interpolate(logits_u_w, (B,1,H,W)[2:], align_corners=True, mode='bilinear')
                        logits_u_w = make_same_size(logits_u_w, s_targets)  
                        #targets_u_w = get_prediction(logits_u_w).unsqueeze(1)
                        pseudo_label = torch.softmax(logits_u_w.detach(), dim=1)
                        max_probs, targets_u_w = torch.max(pseudo_label, dim=1)
                        targets_u_w = targets_u_w.unsqueeze(1)
    
                        # mixing
                        for i in range(len(self.source)*self.train_size):
                            classes = torch.unique(s_targets[i])
                            #classes=classes[classes!=ignore_label]
                            nclasses = classes.shape[0]
                            classes = (classes[torch.Tensor(np.random.choice(nclasses, int((nclasses+nclasses%2)/2),replace=False)).long()]).cuda()              
                            MixMask = self.generate_class_mask(s_targets[i], classes).unsqueeze(0).cuda()
                            mix_img, mix_target = self.strongTransform(MixMask, data=torch.cat((s_imgs[i].unsqueeze(0),t_imgs[i//len(self.source)].unsqueeze(0))), target=torch.cat((s_targets[i].unsqueeze(0),targets_u_w[i//len(self.source)].unsqueeze(0))))
                            if i == 0:
                                mix_imgs = mix_img
                                mix_targets = mix_target
                            else:
                                mix_imgs = torch.cat((mix_imgs, mix_img))
                                mix_targets = torch.cat((mix_targets, mix_target))
                        s_imgs = torch.cat((s_imgs, mix_imgs))
                        s_targets = torch.cat((s_targets, mix_targets))
                        
                    if meta:
                        losses, acc, lr = self.meta_train(epoch - 1, it, to_cuda([s_imgs, s_targets]))
                    else:
                        losses, acc, lr = self.train(epoch - 1, it, to_cuda([s_imgs, s_targets]))
                    if losses != False:
                        loss_meters.update_meters(losses, skips=['dg'] if not meta else [])
                        acc_meters.update_meters(acc)

                        self.print(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta), end='')
                        self.tfb_log(epoch, it, loss_meters, acc_meters)
                        
            self.print(self.train_timer.get_formatted_duration())
            self.log(self.get_string(epoch, it, loss_meters, acc_meters, lr, meta) + '\n')

            if epoch % self.save_interval == 0:
                with self.test_timer:
                    dada_seg_acc = self.val(self.target_loader_val)
                    self.save_best(dada_seg_acc, epoch) 
            self.save('ckpt')
            self.print('Best epoch {} with mIOU {}\n'.format(self.best_target_epoch, self.best_target_acc))
            total_duration = self.train_timer.duration + self.test_timer.duration
            self.print('Time Left : ' + self.train_timer.get_formatted_duration(total_duration * (self.total_epoch - epoch)) + '\n')

        self.log('Best DADA-seg acc : \n  dada_seg : {}, origin : {}, epoch : {}\n'.format(
            self.best_target_acc, self.best_target_acc_source, self.best_target_epoch))
        self.log('Best origin acc : \n  dada_seg : {}, origin : {}, epoch : {}\n'.format(
            self.best_source_acc_target, self.best_source_acc, self.best_source_epoch))

    def save_best(self, dada_seg_acc, epoch):
        self.writer.add_scalar('acc/dada_seg', dada_seg_acc, epoch)
        
        if not self.no_source_test:
            origin_acc = self.val(self.source_val_loader)
            self.writer.add_scalar('acc/origin', origin_acc, epoch)
        else:
            origin_acc = 0
            
        self.writer.flush()
        if dada_seg_acc > self.best_target_acc:
            self.best_target_acc = dada_seg_acc
            self.best_target_acc_source = origin_acc
            self.best_target_epoch = epoch
            self.save('best_dada_seg')

        if origin_acc > self.best_source_acc:
            self.best_source_acc = origin_acc
            self.best_source_acc_target = dada_seg_acc
            self.best_source_epoch = epoch
            self.save('best_origin')

    def val(self, dataset):
        self.backbone.eval()
        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            for p, img, target in dataset:
                img, target = to_cuda(get_img_target(img, target))
                logits = self.backbone(img)
                self.nim(logits, target)
        self.log('\nNormal validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(dataset.dataset, 'format_class_iou'):
            self.log(dataset.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

    def target_specific_val(self, loader):
        self.nim.clear_cache()
        self.nim.set_max_len(len(loader))
        # eval for dropout
        self.backbone.module.remove_dropout()
        self.backbone.module.not_track()
        for idx, (p, img, target) in enumerate(loader):
            if len(img.size()) == 5:
                B, D, C, H, W = img.size()
            else:
                B, C, H, W = img.size()
                D = 1
            img, target = to_cuda([img.reshape(B, D, C, H, W), target.reshape(B, D, 1, H, W)])
            for d in range(img.size(1)):
                img_d, target_d, = img[:, d], target[:, d]
                self.backbone.train()
                with torch.no_grad():
                    new_logits = self.backbone(img_d)
                    self.nim(new_logits, target_d)

        self.backbone.module.recover_dropout()
        self.log('\nTarget specific validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(loader.dataset, 'format_class_iou'):
            self.log(loader.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]

    def target_specific_val_1(self, loader):
        accu = []
        for idx, (p, img, target) in enumerate(loader):
            self.nim.clear_cache()
            self.nim.set_max_len(1)
            # eval for dropout
            self.backbone.module.remove_dropout()
            self.backbone.module.not_track()
            if len(img.size()) == 5:
                B, D, C, H, W = img.size()
            else:
                B, C, H, W = img.size()
                D = 1
            img, target = to_cuda([img.reshape(B, D, C, H, W), target.reshape(B, D, 1, H, W)])
            for d in range(img.size(1)):
                img_d, target_d, = img[:, d], target[:, d]
                self.backbone.train()
                with torch.no_grad():
                    new_logits = self.backbone(img_d)[0]
                    self.nim(new_logits, target_d)
            self.backbone.module.recover_dropout()
            accu.append([idx, self.nim.get_acc()])
        
        accu.sort(reverse=True, key=lambda x:x[1])
        for i in range(10):
            self.log('{} with miou {}\n'.format(accu[i][0], accu[i][1]))

        # self.backbone.module.recover_dropout()
        self.log('\nTarget specific validation : {}\n'.format(self.nim.get_acc()))
        if hasattr(loader.dataset, 'format_class_iou'):
            self.log(loader.dataset.format_class_iou(self.nim.get_class_acc()[0]) + '\n')
        return self.nim.get_acc()[0]


    def predict_target(self, load_path='best_dada_seg', color=False, train=False, output_path='predictions'):
        self.load(load_path)
        import skimage.io as skio
        dataset = self.target_loader_val

        output_path = Path(self.save_path / output_path)
        output_path.mkdir(exist_ok=True)

        if train:
            self.backbone.module.remove_dropout()
            self.backbone.train()
        else:
            self.backbone.eval()

        with torch.no_grad():
            self.nim.clear_cache()
            self.nim.set_max_len(len(dataset))
            # print("Total image:", len(dataset))
            for names, img, target in tqdm(dataset):
                img = to_cuda(img)
                logits = self.backbone(img)[0]
                logits = F.interpolate(logits, img.size()[2:], mode='bilinear', align_corners=True)
                preds = get_prediction(logits).cpu().numpy()
                if color:
                    trainId_preds = preds
                else:
                    trainId_preds = dataset.dataset.predict(preds)

                for pred, name in zip(trainId_preds, names):
                    file_name = name.split('/')[-2] + '_' + name.split('/')[-1]
                    if color:
                        pred = class_map_2_color_map(pred).transpose(1, 2, 0).astype(np.uint8)
                    skio.imsave(str(output_path / file_name), pred)

    def get_string(self, epoch, it, loss_meters, acc_meters, lr, meta):
        string = '\repoch {:4}, iter : {:4}, '.format(epoch, it)
        for k, v in loss_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)
        for k, v in acc_meters.items():
            string += k + ' : {:.4f}, '.format(v.avg)

        string += 'lr : {:.6f}, meta : {}'.format(lr, meta)
        return string

    def log(self, strs):
        self.logger.info(strs)

    def print(self, strs, **kwargs):
        print(strs, **kwargs)

    def tfb_log(self, epoch, it, losses, acc):
        iteration = epoch * len(self.train_loader) + it
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v.val, iteration)
        for k, v in acc.items():
            self.writer.add_scalar('acc/' + k, v.val, iteration)

    def save(self, name='ckpt'):
        info = [self.best_source_acc, self.best_source_acc_target, self.best_source_epoch,
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch]
        dicts = {
            'backbone': self.backbone.state_dict(),
            'opt': self.opt_old.state_dict(),
            'epoch': self.epoch + 1,
            'best': self.best_target_acc,
            'info': info
        }
        self.print('Saving epoch : {}'.format(self.epoch))
        #torch.save(dicts, self.save_path / '{}.pth'.format(name) ,_use_new_zipfile_serialization=False)
        torch.save(dicts, self.save_path / '{}.pth'.format(name) )
 
    def load(self, path=None, strict=False):
        if path is None: 
            path = self.save_path / 'ckpt.pth'
        else:
            if 'pth' in path:
                path = path
            else:
                path = self.save_path / '{}.pth'.format(path)

        try:
            dicts = torch.load(path, map_location='cpu')
            msg = self.backbone.load_state_dict(dicts['backbone'], strict=strict)
            self.print(msg)
            if 'opt' in dicts:
                self.opt_old.load_state_dict(dicts['opt'])
            if 'epoch' in dicts:
                self.epoch = dicts['epoch']
            else:
                self.epoch = 1
            if 'best' in dicts:
                self.best_target_acc = dicts['best']
            if 'info' in dicts:
                self.best_source_acc, self.best_source_acc_target, self.best_source_epoch, \
                self.best_target_acc, self.best_target_acc_source, self.best_target_epoch = dicts['info']
            self.log('Loaded from {}, next epoch : {}, best_target : {}, best_epoch : {}\n'
                     .format(str(path), self.epoch, self.best_target_acc, self.best_target_epoch))
            return True
        except Exception as e:
            self.print(e)
            self.log('No ckpt found in {}\n'.format(str(path)))
            self.epoch = 1
            return False
            
    def create_ema_model(self, model):
        #ema_model = getattr(models, config['arch']['type'])(self.train_loader.dataset.num_classes, **config['arch']['args']).to(self.device)
        ema_model = self.network(**kwargs)
     
        for param in ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(ema_model.parameters())
        n = len(mp)
        for i in range(0, n):
            mcp[i].data[:] = mp[i].data[:].clone()
        #_, availble_gpus = self._get_available_devices(self.config['n_gpu'])
        #ema_model = torch.nn.DataParallel(ema_model, device_ids=availble_gpus)

        ema_model = nn.DataParallel(ema_model)
        return ema_model
        
    def strongTransform(self, mask, data=None, target=None):
        assert ((data is not None) or (target is not None))
        data, target = self.oneMix(mask = mask, data = data, target = target)

        return data, target

    def generate_class_mask(self, pred, classes):
        pred, classes = torch.broadcast_tensors(pred.unsqueeze(0), classes.unsqueeze(1).unsqueeze(2))
        # print(pred.eq(classes).size())
        N = pred.eq(classes).sum(1) 

        return N
    
    def oneMix(self, mask, data = None, target = None):
        #Mix
        if not (data is None):
            stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
            data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0)
        if not (target is None):
            stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
            target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
        return data, target
         
 
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,1,2'
    framework = MetaFrameWork(name='exp', train_num=1, source='GSIM', target='C', debug=False, resume=True)
    framework.do_train()
    framework.val(framework.target_test_loader)
    from eval import test_one_run
    test_one_run(framework, 'previous_exps/dg_all', targets='C', batches=[16, 8, 1], normal_eval=False)
    framework.predict_target()
