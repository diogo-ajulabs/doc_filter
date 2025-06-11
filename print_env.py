import os


def print_environment_variables():
    print("Environment Variables:")
    print("-" * 50)

    # Get all environment variables and sort them alphabetically
    for key, value in sorted(os.environ.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    print_environment_variables()
