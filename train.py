import torch
from modules.utils import resume
from modules.agent import rl_agent_train
from modules.environment import *
from tensorboardX import SummaryWriter
import yaml
import os
from modules.model import *
import torch.nn.functional as F
from data import get_data
import copy
import numpy as np



def train(env, model, config):
    model.to(config['device'])
    model_ast = type(model)(model.input_size, model.n_feachs).to(config['device'])
    cuda = torch.cuda.is_available()

    if os.path.join(os.path.dirname(__file__), config['resume_checkpoint']):
        if os.path.exists(os.path.join(os.path.dirname(__file__), config['resume_checkpoint'])):
            _ = resume(
                model=model,
                cuda=cuda,
                resume_checkpoint=os.path.join(os.path.dirname(__file__), config['resume_checkpoint'])
            )
        else:
            print('checkpoint: "{}" does not exist'.format(config['resume_checkpoint']))
            print('------------------------train from scratch------------------------------------')
    else:
        print('-----------------------------train from scratch------------------------------------')

    if torch.cuda.device_count() >= 1:
        print(" let's use , {} GPU".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model_ast = torch.nn.DataParallel(model_ast)

    model.train()
    model_ast.train(mode=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'], weight_decay=0.0005)
    criterion = torch.nn.MSELoss()

    global_step = 0
    rewards = []
    losses = []

    writer = SummaryWriter(log_dir=os.path.join(os.path.dirname(__file__) + '/runs', config['checkpoint_dir']))
    stats = rl_agent_train(
        model=model,
        model_ast=model_ast,
        env=env,
        step_max=len(env.data) - 1,
        epsilon=config['epsilon'],
        epsilon_min=config['epsilon_min'],
        epsilon_reduce=config['epsilon_reduce'],
        epsilon_reduce_freq=config['epsilon_reduce_freq'],
        device=config['device'],
        memory_size=config['memory_size'],
        global_step=global_step,
        train_freq=config['train_freq'],
        batch_size=config['batch_size'],
        discount_rate=config['discount_rate'],
        criterion=criterion,
        optimizer=optimizer,
        losses=losses,
        rewards=rewards,
        update_model_ast_freq=config['update_model_ast_freq'],
        checkpoint_dir=os.path.join(os.path.dirname(__file__) + '/checkpoint', config['checkpoint_dir']),
        mode=config['mode'],
        writer=writer,
        save_freq=config['save_freq'],
        num_epoch=config['num_epoch'],
        window=config['window']
    )

    return model, stats


def main(config):
    n_feachs = 3
    model = Dueling_Q_Network(64, n_feachs=n_feachs)
    size = 3

    for i in range(0, 20, 2):

        tickers = ["XRP/USD", "ETH/USD", "BTC/USD", "LTC/USD", "LINK/USD", "ADA/USD", "XLM/USD", "XMR/USD"]
        train_data, _ = get_data(ticker=tickers[1], feed_window=1440, prediction_window=0, size=size)

        for episode in range(size):

            env = Environment(data=copy.deepcopy(train_data[episode]))

            model, stats_rsi = train(
                env=env,
                model=model,
                config=config
            )

            if (episode + 1) % config['save_freq'] == 0:
                checkpoint_state = {'epoch': episode, 'state_dict': model.state_dict()}
                torch.save(checkpoint_state,
                           os.path.join(os.path.join(os.path.dirname(__file__) + '/checkpoint', config['checkpoint_dir']), '{}_checkpoint.pth.tar'.format(str(episode) + '_')))


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), 'config/config.yml'), 'r') as stream:
        config = yaml.load(stream)

    print(config)
    main(config)
