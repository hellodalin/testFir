from enviroment import enviroment
import argparse
from Trainer import Trainer
import torch

if __name__=='__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='PPO algorithms with PyTorch for JSP')

    parser.add_argument('--phase', type=str, default='train',
                        help='choose between training phase and testing phase')
    parser.add_argument('--load', type=str, default=None,
                        help='copy & paste the saved model name, and load it')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for random number generators')
    parser.add_argument('--iterations', type=int, default=150000,
                        help='iterations to run and train agent')

    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--input_size', type=int, default=11)
    parser.add_argument('--N', type=int, default=2, help='the number of transformer encoder')
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)

    parser.add_argument('--machine_number', type=int, default=3)
    parser.add_argument('--clip_logits', type=int, default=10,
                        help='improve exploration; clipping logits')

    parser.add_argument('--threshold_return', type=int, default=-230,
                        help='solved requirement for success in given environment')
    parser.add_argument('--tensorboard', action='store_true', default=True)
    parser.add_argument('--device', default=device)
    parser.add_argument('--Initial_jobs', type=int, default=5)
    parser.add_argument('--max_operation_time', type=int, default=100)

    args = parser.parse_args()

    print('开始仿真')
    env = enviroment()
    device = torch.device(args.device)
    train = Trainer(env, device, args)

    # 开始至执行
    env.run()