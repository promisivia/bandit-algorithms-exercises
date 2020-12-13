"""
ex 4.8 "Follow-the-Leader"
chooses each action once and subsequently chooses 
the action with the largest average observed so far
"""
import random


def FollowTheLeader(bandit, n):
    # implement the Follow-the-Leader algorithm
    arm_number = bandit.K
    pull_times = [0] * arm_number  # the pull time of each arm
    rewards = [0] * arm_number  # the total reward of each arm

    # choose each action once
    for a in range(arm_number):
        rewards[a] += bandit.pull(a)
        pull_times[a] += 1

    # choose the action with the largest average
    for t in range(n - arm_number):
        # select arm with max average rewards
        averages = [rewards[i] / pull_times[i] for i in range(arm_number)]
        sets = [index for index, element in enumerate(averages) if element == (max(averages))]
        action = random.choice(sets)

        # pull the chosen arm
        rewards[action] += bandit.pull(action)
        pull_times[action] += 1
