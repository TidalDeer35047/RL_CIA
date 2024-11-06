import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.8
LR = 0.1
LOOP_PENALTY = -10

np.random.seed(42)

class Experience:
    def __init__(self, grid_size):
        self.current = None
        self.replay = []
        self.actions = np.concatenate([np.eye(4, dtype=np.int8), -1 * np.eye(4, dtype=np.int8)], axis=0)
        self.grid_size = grid_size

    def start_new_episode(self):
        if self.current is not None:
            self.replay.append(self.current)
        self.current = None

    def store_set(self, SAR: dict):
        add = np.array(list(SAR.values())).reshape(1, -1)
        if self.current is not None:
            self.current = np.concatenate([self.current, add], axis=0)
        else:
            self.current = add

    def get_latest_reward(self):
        total = np.sum(self.current[:, -1])
        return total

    def get_path(self):
        for path in self.replay:
            turns = path.shape[0]
            total_reward = np.sum(path[:, -1])
            if turns >= total_reward:
                yield path

def check_loop(exp: Experience, next_state, reward):
    if exp.current is not None:
        for ex in exp.current:
            if ex[0] == next_state and ex[-1] > 0:
                print("LOOP FOUND - ", ex, next_state, reward)
                return 1
    return 0

def generate_grid(size=50, num_obstacles=100):
    grid = np.zeros((size, size))
    coord = np.random.randint(0, size, (num_obstacles, 2))
    for obs in coord:
        grid[obs[0], obs[1]] = 1

    # Goal cannot be an obstacle
    grid[0, size - 1] = 0
    plt.imshow(grid, cmap='gray')
    plt.show()
    return grid

def get_reward(state_action: dict, maze, grid_size):
    next_state = state_action["state"] + state_action["action"]
    if np.any(next_state < [0, 0]) or np.any(next_state >= [grid_size, grid_size]):
        return -10, "failed"
    elif maze[int(next_state[0]), int(next_state[1])]:
        return -10, "obstacle"
    elif np.all(next_state == [0, grid_size - 1]):
        return 10, "finished"
    else:
        return 1, "success"

def episode(maze, current_position, experience_buffer: Experience, Q_value, grid_size):
    ACTIONS = experience_buffer.actions
    state = grid_size * current_position[0] + current_position[1]

    for i in range(5000):
        action = np.argmax(Q_value[int(state), :])
        next_state = state + ACTIONS[action]
        next_x, next_y = int(next_state // grid_size), int(next_state % grid_size)
        reward, flag = get_reward({"state": state, "action": action}, maze=maze, grid_size=grid_size)

        if check_loop(experience_buffer, next_state, reward):
            reward = LOOP_PENALTY
            flag = "failed"

        experience_buffer.store_set({"state": state, "action": action, "reward": reward})

        if flag == "finished":
            return 1
        elif flag == "failed":
            current_Q = Q_value[int(state), action]
            next_state_Q = 0
            Q_value[int(state), action] = current_Q + LR * (reward + GAMMA * next_state_Q - current_Q)
            break
        elif flag == "obstacle":
            current_Q = Q_value[int(state), action]
            next_state_Q = 0
            Q_value[int(state), action] = current_Q + LR * (reward + GAMMA * next_state_Q - current_Q)
        else:
            next_state_Q = np.max(Q_value[int(next_state), :])
            current_Q = Q_value[int(state), action]
            Q_value[int(state), action] = current_Q + LR * (reward + GAMMA * next_state_Q - current_Q)
            state = next_state

        if i % 1000 == 0:
            print(f"ITERATION - {i}\n")

    return

def save_to_file(grid_size, num_obstacles, total_episodes, states, Q_value):
    filename = f"navigation_results_modified_ql.txt"

    with open(filename, 'w') as f:
        f.write("Modified Grid Navigation Results\n")
        f.write("=====================\n\n")
        f.write(f"Grid Size: {grid_size}x{grid_size}\n")
        f.write(f"Number of Obstacles: {num_obstacles}\n")
        f.write(f"Total Episodes: {total_episodes}\n\n")

        f.write("Final Path States:\n")
        f.write("================\n")
        for state in states:
            if isinstance(state, dict) and 'state' in state:
                f.write(f"State: {state['state']}, Action: {state['action']}, Reward: {state['reward']}\n")
            else:
                f.write(f"State: {state}\n")

        f.write("\nFinal Q-Values:\n")
        f.write("=============\n")
        for i in range(grid_size):
            for j in range(grid_size):
                state = grid_size * i + j
                f.write(f"\nPosition ({i},{j}):\n")
                for action in range(4):
                    f.write(f"{['Right', 'Up', 'Left', 'Down'][action]}: {Q_value[state, action]:.2f}\n")

    print(f"\nResults have been saved to {filename}")

def main():
    grid_size = int(input("Enter grid size: "))
    obs = int(input("Enter the number of obstructions: "))

    # Generate the grid
    maze = generate_grid(grid_size, obs)
    current_agent_location = np.array([grid_size - 1, 0], dtype=np.int8)
    experience = Experience(grid_size)
    Q_value = np.abs(np.random.normal(0, 5, (grid_size * grid_size, 4)))
    num_episode = 1

    while True:
        status = episode(maze, current_agent_location, experience, Q_value, grid_size)
        if status:
            print("The destination has been reached")
            states = experience.replay[-1]
            # Save results to file
            save_to_file(grid_size, obs, num_episode, states, Q_value)
            break
        total_reward = experience.get_latest_reward()
        print(f"EPISODE {num_episode} REWARD - ", total_reward, end="\n\n")
        num_episode += 1
        experience.start_new_episode()

if __name__ == "__main__":
    main()
