import csv
from ann import ANN

def read(filename):
    fields = []
    rows = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            rows.append(row)
        
    return fields, rows

if __name__ == "__main__":
    f, rows = read("SOCR-HeightWeight.csv")

    nn = ANN(1, [16, 8, 1])
    nn.lr = .001
    nn.layer_activation(-1, "linear")

    data = rows[:1000]
    test_data = rows[-100:]

    inputs = [ [float(item[1])] for item in data]
    labels = [ [float(item[2])] for item in data]

    test_inputs = [ [float(item[1])] for item in test_data]
    test_labels = [ [float(item[2])] for item in test_data]
    # print(inputs[-1])

    sum = 0
    print(f"First record: {inputs[0]}, {labels[0]}")
    print(f"Cost from first record: {nn.cost(inputs[0], labels[0])}")
    print(f"Neuron output: {nn.layers[-1][0].output}")
    for i in range(len(test_data)):
        sum += nn.cost(test_inputs[i], test_labels[i])
    print(f"Total cost from test data before training: {sum}")
    nn.train(inputs, labels, epochs=1)
    print(f"Cost from first record after training: {nn.cost(inputs[0], labels[0])}")
    print(f"Neuron output: {nn.layers[-1][0].output}")
    sum = 0
    # print(f"Test case: {inputs[-1]}, {labels[-1]}")
    for i in range(len(test_data)):
        sum += nn.cost(test_inputs[i], test_labels[i])
    
    print(f"Total cost from test data after training: {sum}")

    