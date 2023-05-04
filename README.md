# cartpole1
## cartpole
* ***목표** : Q-table을 사용하는 Q-learning 에이전트를 이용해서 cartpole 문제 해결*

* **배경지식** 
    - Q-value update  <br>$Q_{t+1}(s_t,a_t)=Q(s_t,a_t)+\alpha _t(s_t,a_t)\times (R_{t+1}+\gamma \times max_aQ_t(s_{t+1},a)-Q(s_t,a_t))$ <br>

    - Q-table을 이용한 Q-learning <br>
    ![](https://wikidocs.net/images/page/165849/Fig_14.png) <br>
    S3이 처음 방문했을 때 이 셀의 Q값은 0이다. 보통 Q-table은 처음에 0으로 채워져있기 때문이다. 위에서 설명한 Q-value update 공식을 사용해 table을 업데이트한다. <br>
    Agent가 다양한 경로를 따라 상태(state)-행동(action) 쌍을 방문하기 시작하면 이전에 0이었던 셀이 채워진다.<br>
    <BR>
    ![](https://wikidocs.net/images/page/165849/Fig_15.png) <br>
    여러 과정을 지나 위 그림과 같이 셀이 채워졌다고 생각해보자. 학습 전 agent는 어떤 action이 다른 action보다 나은지 모른다. 그래서 높은 탐험률을 가지고 새로운 시도를 하게 된다. 하지만 위 그림과 같이 어느정도 Q-table의 셀이 채워지면 S2상태에서 취할 수 있는 action a1,a2,a3,a4 중 가장 Q값이 높은 action a4가 선택된다.

    - Continuous -> Discrete <br>: Q-table은 각각의 state에서 각각의 action이 가지는 Q(s,a)값을 모두 가지고 있다. 하지만 실제 state의 경우 연속적인 값을 가지고 있는 경우가 많기 때문에 무한한 종류의 state를 가질 수 있다. 따라서 Q-table을 만들기 위해선 연속적인 값인 state를 이산값으로 변환시켜주어야 한다. 다음 함수로 continuous한 값을 discrete한 값으로 변환해준다.<br>
    <br>
    ```
    def state_to_bucket(state):
        bucket_indice = []
        for i in range(len(state)):
            if state[i] <= STATE_BOUNDS[i][0]:
                bucket_index = 0
            elif state[i] >= STATE_BOUNDS[i][1]:
                bucket_index = NUM_BUCKETS[i] - 1
            else:
                # Mapping the state bounds to the bucket array
                bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
                offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
                scaling = (NUM_BUCKETS[i]-1)/bound_width
                bucket_index = int(round(scaling*state[i] - offset))
            bucket_indice.append(bucket_index)
    return tuple(bucket_indice)
    ```
    + 위 코드는 주어진 state를 버킷 인덱스로 변환하는 함수이다. 예를 들어, state가 [2.0, 3.5, -1.2, -5.0]이고 NUM_BUCKETS가 [3, 4, 3, 2]이라면 이 함수는 다음과 같은 방식으로 버킷 인덱스를 계산한다. <br>
    <br>    
    1. state 각 요소에 대해, 해당 요소가 버킷의 상한값인 (STATE_BOUNDS[i][1])보다 크거나 같으면 해당 요소의 버킷 인덱스를 NUM_BUCKETS[i]-1 로 설정한다. 또, 버킷의 하한값인 (STATE_BOUNDS[i][0])보다 작거나 같으면 해당 요소의 버킷 인덱스를 0으로 설정한다.

    2. 그 외의 경우, 해당 요소를 STATE_BOUNDS[i]에 대응하는 버킷의 범위로 매핑하여 해당 요소의 버킷 인덱스를 계산한다. 구체적으로, 해당 버킷의 범위를 [STATE_BOUNDS[i][0], STATE_BOUNDS[i][1]]라고 하면, 버킷 인덱스를 계산하기 위해 다음과 같은 공식을 사용한다. <BR>
    
    **bucket_index = int(round(scaling*state[i] - offset))**  
    여기서, scaling은 (NUM_BUCKETS[i]-1)/(STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0])으로 계산된 스케일링 인자이며, offset은 (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/(STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0])으로 계산된 오프셋 값이다. 

    3. 최종적으로, 모든 요소에 대해 계산된 버킷 인덱스를 tuple 형태로 반환한다.
    <br>
* **구현**
1. *Q-table을 위한 상태와 행동 공간 정의*
```
# 상태
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# 행동
NUM_ACTIONS = env.action_space.n # (left, right)
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
ACTION_INDEX = len(NUM_BUCKETS)
```
<BR>

2. *Q-table 초기화 및 탐험률,학습률 등 정의*<br>
+ Q-table은 배열 모두 0을 채움으로써 초기화한다.
+ 탐험률 : 가보지 않은 여러 행동들(action)에 대한 결과를 알기 위해서는 지금 당장 최적이 아니라도 한번쯤 선택해 보아야한다. 만약 탐험률이 5%이면 20번의 행동을 선택할 기회가 있을 떄, 20번 중 1번은 무모한 탐험을 하게 된다.
+ 학습률 : Q-value update 식에서 $\alpha$ 값이 학습률에 해당한다. 현재의 Q값이 제시한 가치와 새로운 경험을 고려한 재귀적 가치를 $\alpha$ 값으로 선형 보강한 중간값으로 Q값을 업데이트하게 된다. 학습률은 0과 1 사이의 값에서 선택하게 된다. 보통 학습의 초반에는 큰 값을 넣어서 새로운 경험에 대한 가중치를 더 주고, 학습이 진행될수록 작은 값을 사용해서 현재의 정책을 신뢰하도록 한다. 
```
#Q-table 초기화
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

MIN_EXPLORE_RATE = 0.01 #최소 탐험률 = 0.01 
MIN_LEARNING_RATE = 0.1 #최소 학습률 = 0.1 

NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = True
```
<BR>

3. *action을 선택하는 함수*
+ 만약  랜덤한 수를 뽑아 (random.random()) 
    * 탐험률보다 작으면 랜덤한 action을 선택 => 탐험(exploration)
    * 탐험률보다 크면 q_table에서 가장 큰 값을 가지는 action을 선택 => 활용(exploitation)
```
def select_action(state, explore_rate):
    
    if random.random() < explore_rate: 
        action = env.action_space.sample()
    
    else: 
        action = np.argmax(q_table[state])

    return action
``` 
<BR>

4. *탐험률(explore rate)를 조절하는 함수*
* get_explore_rate 함수의 입력값 t
    + t >= 24 : 1 과 $1-\log_{10}{t+1\over 25}$ 중 작은 수를 a라고 한다면 a가 최소 탐험률보다 작으면 최소 탐험률을 반환하고 a가 최소 탐험률보다 크면 a값 그대로 반환한다.
    + t < 24 : 1.0을 반환한다.

```
def get_explore_rate(t):
    if t >= 24:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
    else:
        return 1.0
```
<BR>

5. *학습률(learning rate)를 조절하는 함수*
* get_learning_rate 함수의 입력값 t
    + t >= 24 : 1 과 $1-\log_{10}{t+1\over 25}$ 중 작은 수를 a라고 한다면 a가 최소 학습률보다 작으면 최소 학습률을 반환하고 a가 최소 학습률보다 크면 a값 그대로 반환한다.
    + t < 24 : 1.0을 반환한다.
```
def get_learning_rate(t):
    if t >= 24:
         return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
    else:
         return 1.0
```
<BR>

6. *simulate 함수*
+ def simulate():
    * 학습률과 탐험률을 1로 초기화시킨다.
    * Q-value update 공식의 $\gamma$ 에 해당하는 discount_factor 을 0.99로 정의한다. 
    * num_streaks는 이전 승리/패배 기록을 기반으로 현재 연속된 승리/패배 횟수를 나타내는 변수이다. 예를 들어, 만약 이전 5경기의 결과가 (승, 패, 패, 승, 승) 이라면, num_streaks는 현재 연속된 승리 횟수인 2를 나타낸다. 
```
learning_rate = get_learning_rate(0) #1.0 을 return
explore_rate = get_explore_rate(0) #1.0 을 return
discount_factor = 0.99  
num_streaks = 0
```
<BR>

+ episode를 1000번 반복한다.
    + for episode in range(NUM_EPISODES):
        + for t in range(MAX_T): 
```
env.render() #행동을 취하기 이전에 환경에 대해 얻은 관찰값 출력

# 수행할 action 선택
action = select_action(state_0, explore_rate)

# 선택한 action을 수행한 뒤
obv, reward, done, _ = env.step(action)

# 바뀐 state를 bucket으로 변환
state = state_to_bucket(obv)

# 결과에 따라 Q-table 업데이트
best_q = np.amax(q_table[state]) #maxQ'(s',a')
q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

# Setting up for the next iteration
state_0 = state
```
<BR>

+ for t in range(MAX_T) 반복문이 끝났을 때 t값이 미리 정의한 SOLVED_T값인 199 이상이면 num_streaks 를 1 증가시키고 그렇지 않으면 실패로 인해 연속적인 성공을 차단하기 때문에 num_streaks에 0을 대입한다.

```
if done:
    if (t >= SOLVED_T):
        num_streaks += 1
    else:
        num_streaks = 0   
```
<BR>

* for episode in range(NUM_EPISODES) 반복문은 연속으로 성공한 횟수를 나타내는 변수인 num_streaks가 미리 정의해 둔 STREAK_TO_END의 값보다 클 때 종료된다.
```
if num_streaks > STREAK_TO_END:
    break
```
<BR>

+ 다음 step을 위해 parameter update
    * 위에서 정의한 탐험률과 학습률 함수에서 t값 즉 episode가 24 이상일 때 탐험률, 학습률을 미리 정의한 식으로 update시키고 episode 값이 24미만일 때는 무조건 1.0을 return 했으니 23번째 episode까진 무조건 탐험만 하고(탐험률이 1.0 이라는 것은 100% 탐험만 한다는 뜻) 그 이후는 탐험률을 조절한다.
```
explore_rate = get_explore_rate(episode) 
learning_rate = get_learning_rate(episode)
```
<br>

* 전체 코드
```
#cartpole 문제 q-learning으로 풀기
import gym
import numpy as np
import random
import math
from time import sleep



#cartpole 초기 환경 구축
env = gym.make('CartPole-v0')

# Q-table을 위한 상태와 행동 공간 정의
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
ACTION_INDEX = len(NUM_BUCKETS)

# Q-table을 만들기 위해선 연속적인 값인 state를 discrete한 값으로 바꿔줘야함
def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

#Q-table 초기화
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

MIN_EXPLORE_RATE = 0.01 #최소 탐험률 = 0.01 (적어도 0.01으로는 탐험을 해야함)
MIN_LEARNING_RATE = 0.1 #최소 학습률 = 0.1 

NUM_EPISODES = 1000
MAX_T = 250
STREAK_TO_END = 120
SOLVED_T = 199
DEBUG_MODE = True

def simulate():

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0) #1.0 을 return
    explore_rate = get_explore_rate(0) #1.0 을 return
    discount_factor = 0.99  # since the world is unchanging

    num_streaks = 0
    for episode in range(NUM_EPISODES): #1000번 반복

        # 환경 초기화
        obv = env.reset()

        # 초기 state를 bucket으로 변환
        state_0 = state_to_bucket(obv)

        for t in range(MAX_T): #250번 반복
            env.render() #행동을 취하기 이전에 환경에 대해 얻은 관찰값 출력

            # 수행할 action 선택
            action = select_action(state_0, explore_rate)

            # 선택한 action을 수행한 뒤
            obv, reward, done, _ = env.step(action)

            # 바뀐 state를 bucket으로 변환
            state = state_to_bucket(obv)

            # 결과에 따라 Q-table 업데이트
            best_q = np.amax(q_table[state]) #maxQ'(s',a')
            q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if (DEBUG_MODE):
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_streaks)

                print("")

            if done:
               print("Episode %d finished after %f time steps" % (episode, t))
               if (t >= SOLVED_T):
                   num_streaks += 1
               else:
                   num_streaks = 0
               break

            #sleep(0.25)

        # It's considered done when it's solved over 120 times consecutively
        if num_streaks > STREAK_TO_END:
            break

        # 변수를 업데이트
        explore_rate = get_explore_rate(episode) #23번째 episode까진 무조건 탐험만 하고 그 이후는 탐험률을 조절
        learning_rate = get_learning_rate(episode)

#action을 선택하는 함수
def select_action(state, explore_rate):
    
    if random.random() < explore_rate: #랜덤한 수를 뽑아 탐험률보다 작으면 랜덤한 action을 선택
        action = env.action_space.sample()
    
    else: #탐험률보다 크면 q_table에서 가장 큰 값을 가지는 action을 선택
        action = np.argmax(q_table[state])

    return action

#explore rate를 조절하는 함수
def get_explore_rate(t):
    if t >= 24:
        return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))
    else:
        return 1.0

#learning rate를 조절하는 함수
def get_learning_rate(t):
    if t >= 24:
         return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))
    else:
         return 1.0
    
if __name__ == "__main__":
    simulate()
```
<BR>

* *결과* <br>
continous를 discrete으로 바꾸어주는 bucket함수에서 계속 Value error가 발생해 코드가 실행 되지 않았다. 오류 수정을 여러번 시도해 보았으나 해답을 찾지 못해 결국 코드를 완성하지 못하였다. 스터디에서 조언을 구해 다시 오류가 발생한 원인을 제대로 파악하고 코드를 완성해 보아야겠다.


<br>


* *소감* <br>
방학 때 진행한 강화학습에서 실습한 카트폴 문제는 DQN을 사용하여 해결하였다. 카트폴 이후, 케라스에 대해 많이 찾아보았지만 Q-table은 DQN에 비해 상대적으로 지식과 경험이 많이 부족했던 것 같아 Q-table로 카트폴 문제를 해결해 보았다. 아직 애매하게 자리 잡고 있었던 지식을  이전보다 명확하게 정리할 수 있었던 좋은 계기였다. 하지만 역시나 아직 혼자 완전한 코드를 짜기에는 무리가 있었다. 카트폴 외에도 다른 예제들을 참고해 많은 공부가 필요함을 느꼈다. 미숙한 강화학습 지식과 생소한 마크다운으로 생각보다 꽤 많은 시간이 필요했지만 그저 코드를 보고 해석하기 보다 이렇게 글로 정리해 완전히 내 것으로 만드는 과정에서 많은 보람을 느꼈다.

