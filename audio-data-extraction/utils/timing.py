# Source http://stackoverflow.com/questions/1557571/how-to-get-time-of-a-python-program-execution/12344609
# python3 - Windows Version for Linux change clock function for time
import atexit
from time import clock
from datetime import timedelta

# Initialize vars
start = 0.0
end = 0.0


def seconds_to_str(t):
    return str(timedelta(seconds=t))

line = "="*40


def log(s, elapsed=None):
    print(line)
    print(seconds_to_str(clock()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()


def endlog():
    global end
    end = clock()
    elapsed = end-start
    log("End Program", seconds_to_str(elapsed))


def now():
    return seconds_to_str(clock())


def main():
    global start
    start = clock()
    atexit.register(endlog)
    log("Start Program")
