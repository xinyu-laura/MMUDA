import argparse

from TSMLDG import MetaFrameWork

parser = argparse.ArgumentParser(description='TSMLDG train args parser')
parser.add_argument('--name', default='exp', help='name of the experiment')
parser.add_argument('--source', default='CI', help='source domain name list, capital of the first character of dataset "GSIMcuv"(dataset should exists first.)')
parser.add_argument('--target', default='D', help='target domain name, only one target supported')
parser.add_argument('--inner-lr', type=float, default=1e-3, help='inner learning rate of meta update')
parser.add_argument('--outer-lr', type=float, default=5e-3, help='outer learning rate of network update')
parser.add_argument('--resume', action='store_true', help='resume the training procedure')
# parser.add_argument('--debug', action='store_true', help='set the workers=0 and batch size=1 to accelerate debug')
parser.add_argument('--train-size', type=int, default=1, help='the batch size of training')
parser.add_argument('--test-size', type=int, default=2, help='the batch size of evaluation')
parser.add_argument('--train-num', type=int, default=1,
                    help='every ? iteration do one meta train, 1 is meta train, 10000000 is normal supervised learning.')
parser.add_argument('--mix', type=int, default=2, help='0--without mixing, 1--mixing only for meta test, 2--mixing for meta train and test.')
parser.add_argument('--network', default='O', help='R--resnet101, S-Segformer, O--ours.')
parser.add_argument('--opt', default='A', help='S--SGD, A-AdamW')
parser.add_argument('--no-source-test', action='store_false', help='whether test the validation performance in source domain when training')


def train():
    args = vars(parser.parse_args())
    print(args)
    for name in args['source']:
        assert name in 'WABICD'
    assert args['target'][0] in  'WABICD'
    assert len(args['target'])
    framework = MetaFrameWork(**args)
    framework.do_train()


if __name__ == '__main__':
    from utils.task import FunctionJob
    job = FunctionJob([train], gpus=[[1]])
    job.run(minimum_memory=2000)
    # train()