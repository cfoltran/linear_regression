import matplotlib.pyplot as plt
import csv
import sys

def mean(list):
    return sum(list) / len(list)

def var(list):
    return sum((x - mean(list)) ** 2 for x in list) / len(list)

# square root of the variance
def std(list):
    return var(list) ** .5 

# f(x) = ax + b
def model(mileage, t0, t1):
    return (t1 * mileage) + t0

def cost(data, t0, t1):
    m = len(data)
    total_cost = 0

    for mileage, price in data:
        prediction = model(mileage, t0, t1)
        total_cost += (prediction - price) ** 2

    return total_cost / (2 * m)
    
def train(data, learning_rate, n):
    m = len(data)
    theta0, theta1 = 0, 0

    for _ in range(n):
        sum_errors= 0
        sum_errors_mileage = 0
        for mileage, price in data:
            prediction = model(mileage, theta0, theta1)
            # derivate J/a
            sum_errors_mileage += (prediction - price) * mileage
            # derivate J/b
            sum_errors += (prediction - price)
        theta0_tmp = learning_rate * (1 / m) * sum_errors
        theta1_tmp = learning_rate * (1 / m) * sum_errors_mileage

        theta0 -= theta0_tmp
        theta1 -= theta1_tmp

    return theta0, theta1, cost(data, theta0, theta1)

def normalize(data):
    mileages = [item[0] for item in data]
    std_mileages = std(mileages)
    mean_mileages = mean(mileages)

    # z-score normalization
    normalized_data = [((mileage - mean_mileages) / std_mileages, price) for mileage, price in data]
    return normalized_data, mean_mileages, std_mileages

def dernorm_theta(theta0, theta1, mean_mileage, std_mileage):
    dt0 = theta0 - (theta1 * mean_mileage / std_mileage)
    dt1 = theta1 / std_mileage
    return dt0, dt1

def get_dataset():
    try:
        data = []
        with open('data.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            next(reader)
            for [row] in reader:
                data.append((float(row.split(',')[0]), float(row.split(',')[1])))
        return data
    except:
        print("Error: data.csv not found")
        exit()

def main():
    try:
        learning_rate = float(sys.argv[1])
        n = int(sys.argv[2])
        print(learning_rate, n)
        data = get_dataset()
        print(data)
        # x mileage; y price
        normalized_data = normalize(data)

        # Noramalize the data
        normalized_data, mean_mileage, std_mileage = normalize(data)

        # Training
        theta0, theta1, cost = train(normalized_data, learning_rate, n)
        dt0, dt1 = dernorm_theta(theta0, theta1, mean_mileage, std_mileage)
        print('theta0: ', dt0, '\ntheta1: ', dt1, '\ncost: ', cost)

        x = [x for x, _ in data];
        plt.scatter([x for x, _ in data], [y for _, y in data])
        plt.plot([min(x), max(x)], [model(min(x), dt0, dt1), model(max(x), dt0, dt1)], color='red')
        plt.show()
    except:
        print("Usage: python3 train.py [learning_rate] [n_iterations]")


if __name__ == "__main__":
    main()
