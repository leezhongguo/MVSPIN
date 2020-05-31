from utils import TrainOptions
from train import Trainer_li

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    #print(options)
    trainer = Trainer_li(options)
    trainer.train()
