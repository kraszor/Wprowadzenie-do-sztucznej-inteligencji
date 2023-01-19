import numpy as np
from Q_learning import QLearning

TRAIN_ITERATIONS = 10000
TEST_ITERATIONS = int(TRAIN_ITERATIONS / 10)
ENV_NAME = "FrozenLake8x8-v1"
MAX_ACTIONS_PER_ITER = 200


def main():
    train_acc = []
    test_acc = []
    # for var in (0.1, 0.2, 0.3, 0.4):
    for _ in range(25):
        simulator = QLearning(ENV_NAME, 0.99, 0.1, 0.4)
        simulator.init_Q()
        simulator.train(TRAIN_ITERATIONS, MAX_ACTIONS_PER_ITER)
        # print("Accuracy during training")
        accuracy = round(simulator.success/TRAIN_ITERATIONS * 100, 4)
        # print(f'{accuracy}%')
        train_acc.append(accuracy)
        simulator.test(TEST_ITERATIONS, MAX_ACTIONS_PER_ITER)
        # print("\nAccuracy during test")
        accuracy = round(simulator.success/TEST_ITERATIONS * 100, 4)
        # print(f'{accuracy}%')
        test_acc.append(accuracy)
    print("\nTraining stats after 25 attempts")
    mean = np.mean(train_acc)
    std = np.std(train_acc)
    max_ = max(train_acc)
    min_ = min(train_acc)
    print(f"Mean: {mean}    std: {std}  max: {max_}  min: {min_}")
    print("\nTest stats after 25 attempts")
    mean = np.mean(test_acc)
    std = np.std(test_acc)
    max_ = max(test_acc)
    min_ = min(test_acc)
    print(f"Mean: {mean}    std: {std}  max: {max_}  min: {min_}")


if __name__ == "__main__":
    main()
