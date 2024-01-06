import matplotlib.pyplot as plt
import sys
import csv
import train

def main():
    data = train.get_dataset()
    try:
        mileage = float(sys.argv[1])
        theta0 = float(sys.argv[2])
        theta1 = float(sys.argv[3])
        print(mileage, theta0, theta1)

        price = train.model(mileage, theta0, theta1)
        plt.scatter([x for x, _ in data], [y for _, y in data])
        x = [x for x, _ in data];
        plt.plot([min(x), max(x)], [train.model(min(x), theta0, theta1), train.model(max(x), theta0, theta1)], color='green')
        plt.scatter(mileage, price, color='red')
        print('Estimated price: ', price)
        plt.show()
    except:
        print("Usage: python3 predict.py [mileage] [theta0] [theta1]")

if __name__ == "__main__":
    main()
