import numpy as np
from gen_ball import gen_balls
from gbu_tsvm import GBUTSVM

num = 2
purity = 0.96
d1 = 2.0 ** -2
d2 = 2.0 ** 1
du = 1.0
EPS = 0.3

file_path = 'data/blood.csv'  
print(f"Processing {file_path.split('/')[-1]}...")

file_data = np.loadtxt(file_path, delimiter=',')
m, n = file_data.shape
for i in range(m):
    if file_data[i, n - 1] == 0:
        file_data[i, n - 1] = -1

np.random.seed(42)
indices = np.random.permutation(m)
file_data = file_data[indices]

granular_balls = gen_balls(file_data, pur=purity, delbals=num)

Radius = np.array([i[1] for i in granular_balls])
Center = np.array([i[0] for i in granular_balls])
Label = np.array([i[2] for i in granular_balls])

Z_train = np.hstack((Center, Radius.reshape(Radius.shape[0], 1)))
Lab = Label.reshape(Label.shape[0], 1)
A_train_ball = np.hstack((Z_train, Lab))

train_size = int(0.8 * len(file_data))
train_data = file_data[:train_size]
test_data = file_data[train_size:]

try:
    true_labels, predicted_labels, execution_time = GBUTSVM(A_train_ball, test_data, d1, d2, du, EPS)
    accuracy = np.mean(predicted_labels == true_labels) * 100
    
    print(f"Dataset: {file_path.split('/')[-1]}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Time: {execution_time:.4f} seconds")
    print(f"Parameters: num={num}, purity={purity}, d1={d1}, d2={d2}, du={du}, EPS={EPS}")
    print("-" * 50)

except Exception as e:
    print(f"Error processing {file_path.split('/')[-1]}: {str(e)}")

print("Processing completed.")