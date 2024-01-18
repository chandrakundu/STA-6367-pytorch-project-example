# debug_example.py


def divide_numbers(num1, num2):
    result = num2 / num1
    return result


def main():
    a = 0
    b = 10

    result = divide_numbers(a, b)

    print(f"The result of {a} divided by {b} is: {result}")


if __name__ == "__main__":
    main()
