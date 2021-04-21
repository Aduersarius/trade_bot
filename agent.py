import torch
import numpy as np
import os
from utils import shuffle_tensor, policy
from tqdm import tqdm
import torch.nn.functional as F
from data import get_data
from environment import *
def rl_agent_train(model, env, num_epoch, step_max, epsilon, device, memory_size, train_freq, batch_size, mode, model_ast,
                   discount_rate, criterion, optimizer, update_model_ast_freq, epsilon_min, epsilon_reduce_freq, epsilon_reduce,
                   writer, rewards, losses, save_freq, checkpoint_dir, global_step, window):
    for epoch in range(num_epoch):
        state = env.reset()
        total_loss = 0
        total_reward = 0
        total_profit = []
        diff = []
        actions = []
        # add replay memory buffer for the agent
        state_memory = []
        action_memory = []
        reward_memory = []
        observation_memory = []  # observation is actually the next state for boostrap
        for step in tqdm(range(window, step_max)):
            action = policy(model=model, state=state, epsilon=epsilon, device=device)
            observation, stats = env(action)

            state_memory.append(state)
            action_memory.append(action)
            reward_memory.append(stats[0])
            observation_memory.append(observation)
            if len(state_memory) > memory_size:
                state_memory.pop(0)
                action_memory.pop(0)
                reward_memory.pop(0)
                observation_memory.pop(0)

            memory = (state_memory, action_memory, reward_memory, observation_memory)

            if len(state_memory) == memory_size:
                # train or update only in every train freq
                if global_step % train_freq == 0:
                    state_tensor = torch.Tensor(np.array(memory[0], dtype=np.float16)).to(device)
                    action_tensor = torch.Tensor(np.array(memory[1], dtype=np.int16)).to(device)
                    reward_tensor = torch.Tensor(np.array(memory[2], dtype=np.int16)).to(device)
                    observation_tensor = torch.Tensor(np.array(memory[3], dtype=np.float16)).to(device)
                    #print(state_tensor.shape,"tensor")
                    shuffle_index = shuffle_tensor(memory_size, device)
                    shuffle_state = torch.index_select(state_tensor, 0, shuffle_index)
                    shuffle_reward = torch.index_select(reward_tensor, 0, shuffle_index)
                    shuffle_action = torch.index_select(action_tensor, 0, shuffle_index)
                    shuffle_observation = torch.index_select(observation_tensor, 0, shuffle_index)
                    #print(shuffle_state.shape,"shuffle")
                    for i in range(memory_size)[::batch_size]:
                        batch_state = shuffle_state[i: i + batch_size, :].type('torch.FloatTensor').to(device)
                        batch_action = shuffle_action[i: i + batch_size].type('torch.FloatTensor').to(device)
                        batch_reward = shuffle_reward[i: i + batch_size].type('torch.FloatTensor').to(device)
                        batch_observation = shuffle_observation[i: i + batch_size, :].type('torch.FloatTensor').to(
                            device)

                        q_eval = model(batch_state).gather(1, batch_action.long().unsqueeze(1))
                        q_next = model_ast(batch_observation).detach()

                        if mode == 'dqn':
                            q_target = batch_reward + discount_rate * q_next.max(1)[0]

                        elif mode == 'ddqn':
                            q_target = batch_reward + discount_rate * q_next.gather(1, torch.argmax(
                                model(batch_observation), dim=1, keepdim=True))

                        else:
                            raise ValueError('please input correct mode for rl agent, either "dqn", or "ddqn"')

                        optimizer.zero_grad()
                        #print(q_eval.view(-1, 1).shape, q_target[0].shape)
                        loss = criterion(q_eval, q_target[0].view(-1, 1))
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                        

                if global_step % update_model_ast_freq == 0:
                    para = {k: 0.3 * v + 0.7 * model.state_dict()[k] for k, v in model_ast.state_dict().items()}
                    model_ast.load_state_dict(para)

            # epsilon
            if epsilon > epsilon_min and global_step % epsilon_reduce_freq == 0:
                epsilon -= epsilon_reduce

            total_reward += stats[0]
            state = observation
            global_step += 1
            # here writer is tensorboardX
        
        print(
            '--------------------------------------- epoch {}/{} ----------------------------------------'.format(
                epoch+1, num_epoch))
        print('total_reward : {}'.format(total_reward))
        print('total_loss : {}'.format(total_loss))
        print("Balance this epoch :", "{:.2%}".format(stats[3]-1), ' | Diff:', "{:.2%}".format(stats[3]-stats[4]),
        ' | Actions:', stats[2])
        

        rewards.append(total_reward)
        losses.append(total_loss)
        total_profit.append(stats[1])
        actions.append(stats[2])
        diff.append(stats[3]-stats[4])

        # if writer:
        #     writer.add_scalar('diff', stats[3]-stats[4], global_step=global_step)
        # else:
        #     print("no writer!")
        # if (epoch + 1) % save_freq == 0:
        #     checkpoint_state = {'epoch': epoch, 'state_dict': model.state_dict()}
        #     torch.save(checkpoint_state, os.path.join(checkpoint_dir, '{}_checkpoint.pth.tar'.format(str(epoch) + '_' + str(global_step))))


        # if writer:
        #     writer.add_scalar('loss', total_loss, global_step=global_step)
        #     writer.add_scalar('reward', total_reward, global_step=global_step)
        #     writer.add_scalar('profit', total_profit, global_step=global_step)


    stats_eval = eval(model=model)
    stats[0] = rewards
    stats[1] = (((sum(total_profit)/len(total_profit))-1)*100)
    stats[2] = sum(actions)/len(actions)
    stats.append(sum(losses)/len(losses))
    stats.append(((sum(diff)/len(diff))-1)*100)
    stats.extend(stats_eval)

    return stats


def eval(model, size=10, ticker="ETH/USD"):
    data, _ = get_data(ticker=ticker, feed_window=1440, prediction_window=0, size=size)
    total_profit = []
    diff = []
    actions = []
    for episode in range(len(data)):
        env = Environment(data=copy.deepcopy(data[episode]))
        observation = env.reset()
        for step in range(64, len(env.data) - 1):
            action = model(torch.Tensor(np.array(observation, dtype=np.float16)).view(-1,1, 3).to(torch.device("cuda")))
            action = action.data.argmax()
            observation, stats = env(action)
        total_profit.append(stats[1])
        actions.append(stats[2])
        diff.append(stats[3]-stats[4])
    print('-----------------------------------------------------------------------------------------')
    print("Profit eval:", "{:.2%}".format((sum(total_profit)/len(total_profit))-1), ' | Diff eval:', "{:.2%}".format(sum(diff)/len(diff)),
        ' | Actions eval:', sum(actions)/len(actions))
    print('-----------------------------------------------------------------------------------------')
    return [(sum(total_profit)/len(total_profit))-1, sum(diff)/len(diff)]

        




