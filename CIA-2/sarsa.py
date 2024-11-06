import numpy as np
import matplotlib.pyplot as plt

GAMMA = 0.8  # Discount factor
ALPHA = 0.1  # Learning rate
EPSILON = 0.2  # Exploration rate

np.random.seed(56)

class Experience:
    def __init__(self):
        self.current = None
        self.replay = []
        self.actions = np.concatenate([np.eye(4), -np.eye(4)], axis=0)  # 4 actions: up, down, left, right

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
        return 1

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

def get_action(q_values, epsilon=EPSILON):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 4)  # Random action
    else:
        return np.argmax(q_values)  # Greedy action

def get_reward(state_action: dict, maze, grid_size):
    next_state = state_action["state"] + state_action["action"]
    if np.any(next_state < [0, 0]) or np.any(next_state >= [grid_size, grid_size]):
        return -10, "failed"
    elif maze[next_state[0], next_state[1]]:
        return -10, "obstacle"
    elif np.all(next_state == [0, grid_size - 1]):
        return 10, "finished"
    else:
        return 1, "success"

def episode(maze, current_position, experience_buffer: Experience, q_table, grid_size):
    ACTIONS = np.concatenate([np.eye(2, dtype=np.int8), -np.eye(2, dtype=np.int8)], axis=0, dtype=np.int8)
    state = grid_size * current_position[0] + current_position[1]

    while True:
        action = get_action(q_table[state, :])
        next_state = current_position + ACTIONS[action]
        x, y = next_state

        reward, flag = get_reward(
            {"state": state, "action": action},
            maze=maze,
            grid_size=grid_size
        )

        if check_loop(experience_buffer, grid_size * x + y, reward):
            reward = -5
            flag = "failed"

        experience_buffer.store_set({"state": state, "action": action, "reward": reward})

        if flag == "finished":
            return 1
        elif flag == "failed":
            q_table[state, action] = q_table[state, action] + ALPHA * (reward + GAMMA * np.max(q_table[grid_size * x + y, :]) - q_table[state, action])
            break
        elif flag == "obstacle":
            q_table[state, action] = q_table[state, action] + ALPHA * (reward + GAMMA * 0 - q_table[state, action])
            next_state = None
        else:
            q_table[state, action] = q_table[state, action] + ALPHA * (reward + GAMMA * np.max(q_table[grid_size * x + y, :]) - q_table[state, action])

        if next_state is None:
            current_position = current_position
        else:
            current_position = next_state
        state = grid_size * current_position[0] + current_position[1]

    return

def save_to_file(grid_size, num_obstacles, total_episodes, total_reward, q_table):
    filename = f"navigation_results_modified_sarsa.txt"
    with open(filename, 'w') as f:
        f.write("Grid Navigation Results\n")
        f.write("=====================\n\n")
        f.write(f"Grid Size: {grid_size}x{grid_size}\n")
        f.write(f"Number of Obstacles: {num_obstacles}\n")
        f.write(f"Total Episodes: {total_episodes}\n")
        f.write(f"Final Total Reward: {total_reward}\n\n")
        f.write("Final Q-Values:\n")
        f.write("=============\n")
        for i in range(grid_size):
            for j in range(grid_size):
                f.write(f"\nPosition ({i},{j}):\n")
                f.write(f"Up: {q_table[grid_size * i + j, 0]:.2f}\n")
                f.write(f"Down: {q_table[grid_size * i + j, 1]:.2f}\n")
                f.write(f"Left: {q_table[grid_size * i + j, 2]:.2f}\n")
                f.write(f"Right: {q_table[grid_size * i + j, 3]:.2f}\n")

    print(f"\nResults have been saved to {filename}")

def main():
    grid_size = int(input("Enter grid size: "))
    obs = int(input("Enter the number of obstructions: "))

    # Generate the grid
    maze = generate_grid(grid_size, obs)
    current_agent_location = np.array([grid_size - 1, 0], dtype=np.int8)
    experience = Experience()
    q_table = np.zeros((grid_size * grid_size, 4))
    num_episode = 1

    while True:
        status = episode(maze, current_agent_location, experience, q_table, grid_size)
        if status:
            print("The destination has been reached")
            total_reward = experience.get_latest_reward()
            # Save results to file
            save_to_file(grid_size, obs, num_episode, total_reward, q_table)
            break
        total_reward = experience.get_latest_reward()
        print(f"EPISODE {num_episode} REWARD - ", total_reward, end="\n\n")
        num_episode += 1
        experience.start_new_episode()

if __name__ == "__main__":
    main()
