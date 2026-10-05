# exercise 1 - Python virtual environment setup and test
# @authors: nobodycaresdude with help from Claude Opus 5.5
# simple Python application to test Docker setup

import sys
print(f"Hello, from docker!")
if len(sys.argv) > 1:
    print(f"The parameter I received is: {sys.argv[1]}")