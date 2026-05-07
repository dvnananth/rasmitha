def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b


if __name__ == "__main__":
    x = int(input("Enter first number: "))
    y = int(input("Enter second number: "))

    result = add_numbers(x, y)
    print("Sum =", result)