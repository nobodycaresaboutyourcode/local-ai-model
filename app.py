import sys
print(f"Hello, from docker!")
if len(sys.argv) > 1:
    print(f"The parameter I received is: {sys.argv[1]}")