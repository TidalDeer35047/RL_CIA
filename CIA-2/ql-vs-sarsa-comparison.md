**Q-Learning vs SARSA**

1. **Convergence Properties**:
   - Q-Learning is an off-policy algorithm, which means it can learn the optimal policy even if the agent is following a different (greedy) policy. This makes Q-Learning more robust to exploration strategies.
   - SARSA is an on-policy algorithm, which means it learns the value of the policy the agent is currently following. This can make SARSA more sensitive to the exploration/exploitation tradeoff.


2. **Handling Loops**:
   - The modified SARSA implementation includes a specific check for loops and applies a penalty to discourage revisiting the same states. This can help the agent learn to avoid getting stuck in loops.
   - The Q-Learning implementation does not have an explicit loop-handling mechanism, which could potentially lead to the agent getting stuck in loops more often.

4. **Computational Complexity**:
   - Both algorithms have similar computational complexity, as they both need to update the Q-table for each step in the episode.
