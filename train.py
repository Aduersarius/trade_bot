import torch
from modules.utils import resume
from modules.agent import rl_agent_train
from modules.environment import *
from tensorboardX import SummaryWriter
import yaml
import os
from modules.model import Dueling_Q_Network
from dataloader.dataset import read_data
from data import get_data
import copy
import numpy as np


def train(env, model, config):
    model.to(config['device'])
    model_ast = type(model)(model.input_size).to(config['device'])
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

    if torch.cuda.device_count() > 1:
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
    model = Dueling_Q_Network(64)
    size = 100
    profit, profit_b, diff_, a = [], [], [], []
    for i in range(0, 20, 2):
        profits = []
        diff = []
        profits2, act = [], []
        tickers = ["XRP/USD", "ETH/USD", "BTC/USD", "LTC/USD", "LINK/USD", "ADA/USD", "XLM/USD", "XMR/USD"]
        for ticker in tickers:
            train_data, _ = get_data(ticker=ticker, feed_window=1440, prediction_window=0, size=size)
            # test_data, _ = get_data(ticker="AAPL", feed_window=390, prediction_window=0, size=size)
            # train_data, test_data = read_data(
            #     path=os.path.join(os.path.dirname(__file__), config['data_name']),
            #     start_date=config['start_date'],
            #     split_date=config['split_date']
            # )

            for episode in range(size):
                try:
                    rsi_env = Environment_rsi(data=copy.deepcopy(train_data[episode]), i=i)
                    # supertrend_env = Environment_supertrend(data=copy.deepcopy(train_data[episode]), history_length=64)

                    # test_env = Environment(data=test_data[episode], history_length=50)

                    # model, stats_st = train(
                    #     env=supertrend_env,
                    #     model=model,
                    #     config=config
                    # )
                    model, stats_rsi = train(
                        env=rsi_env,
                        model=model,
                        config=config
                    )
                    print("Profit rsi:", "{:.2%}".format(stats_rsi[1] - 1), "  Actions:", stats_rsi[2])
                    print("Buy&hold:", "{:.2%}".format(stats_rsi[3]), "  Loss:", "{:.2%}".format(stats_rsi[3]),
                          "  diff:",
                          "{:.2%}".format(stats_rsi[5] - 1), " Profits balance:", "{:.2%}".format(stats_rsi[4]))
                    profits.append(stats_rsi[1] - 1)
                    diff.append(stats_rsi[5] - 1)
                    profits2.append(stats_rsi[4])
                    act.append(stats_rsi[2])

                except Exception:
                    pass
        print("------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------")
        print("---------------- Profit:", "{:.2%}".format(np.mean(profits)), "-------------------")
        print("---------------- Profit with balance:", "{:.2%}".format(np.mean(profits2)), "-------------------")
        print("----------------  Diff:", "{:.2%}".format(np.mean(diff)), "-------------------")
        print("------------------------------------------------------------------------------------")
        profit.append(np.mean(profits))
        profit_b.append(np.mean(profits2))
        diff_.append(np.mean(diff))
        a.append(np.mean(act))

    for i in range(len(profit)):
        print("========================================================================")
        print(40 - 2 * i, "и", 60 + 2 * i, "|| profit:", "{:.2%}".format(profit[i]), "|| profit balance:",
              "{:.2%}".format(profit_b[i]), "|| diff:", "{:.2%}".format(diff_[i]), "|| actions:", a[i])
        # if (episode + 1) % config['save_freq'] == 0:
        #     checkpoint_state = {'epoch': episode, 'state_dict': model.state_dict()}
        #     torch.save(checkpoint_state,
        #                os.path.join(os.path.join(os.path.dirname(__file__) + '/checkpoint', config['checkpoint_dir']), '{}_checkpoint.pth.tar'.format(str(episode) + '_')))
    return


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), 'config/config.yml'), 'r') as stream:
        config = yaml.load(stream)

    print(config)
    _ = main(config)
