<h1 align=center>DQN-Breakout with pytorch</h1>


## Ⅰ  Description of the Breakout Game

> In a Breakout game:
>
> - A player is given a paddle that it can move horizontally
> - At the beginning of each turn, a ball drops down automatically from somewhere in the screen*
> - The paddle can be used to bounce back the ball
> - There are layers of bricks in the upper part of the screen
> - The player is awarded to destroy as many bricks as possible by hitting the bricks with the bouncy ball
> - The player is given 5 turns in each game

##  Ⅱ Our Assignment

- Read through the implementation and explain in detail in your team report what each component is responsible for and how the components are connected together.
- Besides, in this report, we are committed to researching the **third** question which is listed in the powerpoint: **Stabilize the movement of the paddle (avoid high-freq paddle shaking effects) so that the agent plays more like a human player**.
- Then, we open-source our project in GitHub.

## Ⅲ The Explanation the Implementation

### 1. main.py

- It contains the simplest ddqn process. 

- In each iteration, the agent selects an action：

    ```python
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    ```

-  Then the environment executes the action, and stores the relevant reward information in memory:

    ```python
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done)
    ```

- And according to a certain frequency, learn from memory, or synchronize two q-tables:

    ```python
    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)
    if step % TARGET_UPDATE == 0:
        agent.sync()
    ```

- Also output reward and save model at a certain frequency：

    ```python
    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
    ```

### 2. utils_drl.py

- It contains a class called `Agent`. `Agent` has the concrete realization of each step of reinforcement learning

- The first is the `run` function, which implements the e-greedy strategy

    ```python
    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)
    
        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)
    ```

- Then there's the `learn` function. It takes a batch data from memory and calculates the loss to update the network. DDQN calculates loss as follows:

    - Use current network $(\mathbf{w})$ to select actions
    - Use older network $\left(\mathbf{w}^{-}\right)$ to evaluate actions
    - $\left(r+\gamma Q\left(s^{\prime}, \arg \max _{a^{\prime}} Q\left(s^{\prime}, a^{\prime}, \mathbf{w}\right), \mathbf{w}^{-}\right)-Q(s, a, \mathbf{w})\right)^{2}$

    The code is shown below:

    ```python
    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size)
    
        values = self.__policy(state_batch.float()).gather(1, action_batch)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)
    
        self.__optimizer.zero_grad()
        loss.backward()
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)
            self.__optimizer.step()
    
            return loss.item()
    ```

- Next is the `sync` function for synchronizing q-table and the `save` function for saving the model

    ```python
    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target network."""
        self.__target.load_state_dict(self.__policy.state_dict())
    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
    ```

### 3. utils_env.py

- This code is mainly used to interact with the breakout game environment

- The first is the `reset` function, which is used to reset the environment when the game is over:

    ```python
    def reset(
            self, render: bool = False,
    ) -> Tuple[List[TensorObs], float, List[GymImg]]:
        """reset resets and initializes the underlying gym environment."""
        self.__env.reset()
        init_reward = 0.
        observations = []
        frames = []
        for _ in range(5): # no-op
            obs, reward, done = self.step(0)
            observations.append(obs)
            init_reward += reward
            if done:
                return self.reset(render)
            if render:
                frames.append(self.get_frame())
        return observations, init_reward, frames
    ```

- Then there is the `step` function, which calculates the next step of the game for an input action

    ```python
    def step(self, action: int) -> Tuple[TensorObs, int, bool]:
        """step forwards an action to the environment and returns the newest
        observation, the reward, and an bool value indicating whether the episode is terminated."""
        action = action + 1 if not action == 0 else 0
        obs, reward, done, _ = self.__env.step(action)
        return self.to_tensor(obs), reward, done
    ```

- `get_ frame` function is used to get a screenshot of the game. The screenshot will be entered into the network as a state

    ```python
    def get_frame(self) -> GymImg:
        """get_frame renders the current game frame."""
        return Image.fromarray(self.__env.render(mode="rgb_array"))
    ```

- Then there are some functions for format conversion and getting constants and so on.

    ```python
    @staticmethod
    def to_tensor(obs: GymObs) -> TensorObs:
        """to_tensor converts an observation to a torch tensor."""
        return torch.from_numpy(obs).view(1, 84, 84)
    
    @staticmethod
    def get_action_dim() -> int:
        """get_action_dim returns the reduced number of actions."""
        return 3
    
    @staticmethod
    def get_action_meanings() -> List[str]:
        """get_action_meanings returns the actual meanings of the reduced actions."""
        return ["NOOP", "RIGHT", "LEFT"]
    
    @staticmethod
    def get_eval_lives() -> int:
        """get_eval_lives returns the number of lives to consume in an evaluation round."""
        return 5
    
    @staticmethod
    def make_state(obs_queue: deque) -> TensorStack4:
        """make_state makes up a state given an obs queue."""
        return torch.cat(list(obs_queue)[1:]).unsqueeze(0)
    
    @staticmethod
    def make_folded_state(obs_queue: deque) -> TensorStack5:
        """make_folded_state makes up an n_state given an obs queue."""
        return torch.cat(list(obs_queue)).unsqueeze(0)
    
    @staticmethod
    def show_video(path_to_mp4: str) -> None:
        """show_video creates an HTML element to display the given mp4 video in IPython."""
        mp4 = pathlib.Path(path_to_mp4)
        video_b64 = base64.b64encode(mp4.read_bytes())
        html = HTML_TEMPLATE.format(alt=mp4, data=video_b64.decode("ascii"))
        ipydisplay.display(ipydisplay.HTML(data=html))
    ```

- Finally, there is an `evaluate` function, which can calculate the average reward of several games. It is used to output the current reward that can be run according to a certain frequency during training

    ```python
    def evaluate(
            self,
            obs_queue: deque,
            agent: Agent,
            num_episode: int = 3,
            render: bool = False,
    ) -> Tuple[
        float,
        List[GymImg],
    ]:
        """evaluate uses the given agent to run the game for a few episodes and
        returns the average reward and the captured frames."""
        self.__env = self.__env_eval
        ep_rewards = []
        frames = []
        for _ in range(self.get_eval_lives() * num_episode):
            observations, ep_reward, _frames = self.reset(render=render)
            for obs in observations:
                obs_queue.append(obs)
            if render:
                frames.extend(_frames)
            done = False
            while not done:
                state = self.make_state(obs_queue).to(self.__device).float()
                action = agent.run(state)
                obs, reward, done = self.step(action)
                ep_reward += reward
                obs_queue.append(obs)
                if render:
                    frames.append(self.get_frame())
            ep_rewards.append(ep_reward)
        self.__env = self.__env_train
        return np.sum(ep_rewards) / num_episode, frames
    ```

### 4. utils_memory.py

- This module implements two functions: putting experience into memory and extracting some experience from it.

- The `push` function puts experience into it:

    ```python
    def push(
        self,
        folded_state: TensorStack5,
        action: int,
        reward: int,
        done: bool,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)
    ```

- The `sample` function takes the experience out. Here is random access. If you want to implement prioritized experience replay, you must start with this function

    ```python
    def sample(self, batch_size: int) -> Tuple[
        BatchState,
        BatchAction,
        BatchReward,
        BatchNext,
        BatchDone,
    ]:
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done
    ```

### 5. utils_model.py

This module is a neural network for image processing. The input is an image, i.e. state, which can output Q values corresponding to different actions. It is equivalent to a q-table function

```python
class DQN(nn.Module):
    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)
        self.__device = device
    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)
    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
```

### 6. utils_types.py

This file defines some type names. It doesn't really work, it's just easy to use when we program.

```python
"""
Aliases created in this module are useless for static type checking, instead,
they act as hints for human only
"""
from typing import (
    Any,
)
# Tensor with shape (None, 4, 84, 84)
BatchState = Any
# Tensor with shape (None, 1)
BatchAction = Any
# Tensor with shape (None, 1)
BatchReward = Any
# Tensor with shape (None, 4, 84, 84)
BatchNext = Any
# Tensor with shape (None, 1)
BatchDone = Any
# NDArray with shape (210, 160, 3)
GymImg = Any
# NDArray with shape (84, 84, 1)
GymObs = Any
# Tensor with shape (N, 1)
TensorN1 = Any
# Tensor with shape (1, 84, 84)
TensorObs = Any
# A stack with 4 GymObs, with shape (1, 4, 84, 84)
TensorStack4 = Any
# A stack with 5 GymObs, with shape (1, 5, 84, 84)
TensorStack5 = Any
# torch.device("cpu") or torch.device("cuda"), can be conditional on
# torch.cuda.is_available()
TorchDevice = Any
```

## Ⅳ The Implementation of the Question3

- The problem is to stabilize the movement of the paddle so that the agent plays more like a human player

- At first, we naturally thought that we could add a penalty for jitter or a reward for stillness. But we also feel that this is a temporary solution rather than a permanent solution.

- Is it a direct way to control it simply by reward? Or, where did he learn this bad habit of shaking? So we realized that this kind of jitter was probably learned in the process of random exploration in the beginning.

- Firstly, the way we think of is to add a limit to the random process to keep the last action with a probability of 0.85. Because the last action is also random, we don't think it will affect the randomness. To achieve this, in `main.py` We will pass in the value of the last action `preaction` to the `run` function

    ```python
    action = agent.run(state, training, preaction)
    preaction = action
    ```

- And then in `utils_ drl.py`, modify the `run` function in the `Agent` class.

    ```python
    def run(self, state: TensorStack4, training: bool = False, preaction: int = 0) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)
        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        if self.__r.random() > 0.15: #keep the last action with a probability of 0.85
                return preaction
        return self.__r.randint(0, self.__action_dim - 1)
    ```

- This implementation has some effect with less high-freq paddle shaking, but it is not particularly good. So we have a new promotion.

- First, let's see the movement which is not been improved:

    <video src="https://www.hz-heze.com/wp-content/uploads/2020/11/3.2.1.mp4"></video>

    > **(If you open this report with Chrome or Microsoft Edge, you may see a static picture up, but it's a video indeed, you just need to Right-click it and click "显示控件", then the video will play automatically.)**
    >
    > <img src="https://www.hz-heze.com/wp-content/uploads/2020/11/截屏2020-11-14-22.20.57.png" alt="1" style="zoom: 45%;" />

We can see that it isn't stable at all.

Then, let's see the movement now:

<video src="https://www.hz-heze.com/wp-content/uploads/2020/11/3.2.2.mp4"></video>

It indeed has some improvement, but it's not as stable as we want it to be.

- We found that not shaking means paddle stays in a place, not keep the last move. So we decided to keep paddle in a place with a probability of 0.85 in a random process.

    ```python
    def run(self, state: TensorStack4, training: bool = False, preaction: int = 0) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)
        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        if self.__r.random() > 0.15:
                return int(0)  #keep stay in a place with a probability of 0.85
        return self.__r.randint(0, self.__action_dim - 1)
    ```

    This time, the effect is much better:

<video src="https://www.hz-heze.com/wp-content/uploads/2020/11/3_model_262.mp4"></video>

Obviously, it's very stable. 






