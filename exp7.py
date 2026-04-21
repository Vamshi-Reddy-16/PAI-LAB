import numpy as np
class GridWorldMDP:
    def __init__(self, size, goal, trap):
        self.size = size
        self.goal = goal
        self.trap = trap
        self.state_space = [(i, j) for i in range(size) for j in range(size)]
        self.action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.transitions = self.build_transitions()
        self.rewards = self.build_rewards()
    def build_transitions(self):
        transitions = {}
        for state in self.state_space:
            transitions[state] = {}
            for action in self.action_space:
                transitions[state][action] = self.calculate_transitions(state, action)
        return transitions
    def calculate_transitions(self, state, action):
        i, j = state
        if action == 'UP':
            next_state = self.validate_state(i - 1, j)
        elif action == 'DOWN':
            next_state = self.validate_state(i + 1, j)
        elif action == 'LEFT':
            next_state = self.validate_state(i, j - 1)
        elif action == 'RIGHT':
            next_state = self.validate_state(i, j + 1)
        return [(1.0, next_state)]
    def validate_state(self, i, j):
        i = max(0, min(i, self.size - 1))
        j = max(0, min(j, self.size - 1))
        return (i, j)
    def build_rewards(self):
        rewards = {}
        for state in self.state_space:
            rewards[state] = -1.0
        rewards[self.goal] = 0.0
        rewards[self.trap] = -10.0
        return rewards
def value_iteration(mdp, gamma=0.9, epsilon=0.01):
    state_values = {state: 0.0 for state in mdp.state_space}
    iteration = 0
    while True:
        iteration += 1
        delta = 0
        for state in mdp.state_space:
            if state == mdp.goal or state == mdp.trap:
                continue
            v = state_values[state]
            new_value = max([
                sum([
                    p * (mdp.rewards[next_state] + gamma * state_values[next_state])
                    for p, next_state in mdp.transitions[state][action]
                ])
                for action in mdp.action_space
            ])
            state_values[state] = new_value
            delta = max(delta, abs(v - new_value))
        print(f"\n--- Value Iteration: Iteration {iteration} | Max Delta: {round(delta, 4)} ---")
        for s, v in state_values.items():
            print(f"  {s} : {round(v, 2)}")
        if delta < epsilon:
            print(f"\n  [Converged at Iteration {iteration}]")
            break
    return state_values
def policy_iteration(mdp, gamma=0.9):
    policy = {
        state: np.random.choice(mdp.action_space)
        for state in mdp.state_space
        if state != mdp.goal and state != mdp.trap
    }
    state_values = {state: 0.0 for state in mdp.state_space}
    outer_iteration = 0
    while True:
        outer_iteration += 1
        eval_iteration = 0
        while True:
            eval_iteration += 1
            delta = 0
            for state in mdp.state_space:
                if state == mdp.goal or state == mdp.trap:
                    continue
                v = state_values[state]
                action = policy[state]
                new_value = sum([
                    p * (mdp.rewards[next_state] + gamma * state_values[next_state])
                    for p, next_state in mdp.transitions[state][action]
                ])
                state_values[state] = new_value
                delta = max(delta, abs(v - new_value))
            print(f"\n--- Policy Iteration {outer_iteration} | Evaluation Step {eval_iteration} | Max Delta: {round(delta, 4)} ---")
            for s, v in state_values.items():
                print(f"  {s} : {round(v, 2)}")
            if delta < 0.01:
                print(f"  [Evaluation converged at step {eval_iteration}]")
                break
        policy_stable = True
        print(f"\n  [Policy Improvement — Iteration {outer_iteration}]")
        for state in mdp.state_space:
            if state == mdp.goal or state == mdp.trap:
                continue
            old_action = policy[state]
            best_action = max(
                mdp.action_space,
                key=lambda a: sum([
                    p * (mdp.rewards[next_state] + gamma * state_values[next_state])
                    for p, next_state in mdp.transitions[state][a]
                ])
            )
            policy[state] = best_action
            if old_action != best_action:
                print(f"    {state}: {old_action} -> {best_action}  (policy updated)")
                policy_stable = False
        if policy_stable:
            print(f"\n  [Policy stable — converged at outer iteration {outer_iteration}]")
            break
    return policy, state_values
size = 3
goal = (2, 2)
trap = (1, 1)
mdp = GridWorldMDP(size, goal, trap)
print("=" * 55)
print("VALUE ITERATION")
print("=" * 55)
vi_values = value_iteration(mdp)
print("\n" + "=" * 55)
print("POLICY ITERATION")
print("=" * 55)
policy, pi_values = policy_iteration(mdp)
print("\n" + "=" * 55)
print("FINAL POLICY ITERATION RESULTS")
print("=" * 55)
for s in policy:
    print(f"  {s} -> {policy[s]} | Value: {round(pi_values[s], 2)}")
