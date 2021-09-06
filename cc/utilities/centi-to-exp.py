import argparse
import math

def lichess_model(cp):
    return round(1000 / (1 + math.exp(-0.004 * cp)))

def out(cp):
    w = lichess_model(cp)
    return (w/1000)*2 - 1

def delta(before, after, to_move):
    sign = 1 if to_move == 'w' else -1
    return (out(before) - out(after))*sign


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate the exp-value equivalent to a given centipawn delta.')
    parser.add_argument('-a', metavar='num', type=float, help='the first value in centipawns', required=True)
    parser.add_argument('-b', metavar='num', type=float, help='the second value in centipawns', required=True)
    parser.add_argument('-c', metavar='colour', type=str, help='the colour of side to move, w or b', required=True)
    args = parser.parse_args()
    print(delta(args.a, args.b, args.c))
