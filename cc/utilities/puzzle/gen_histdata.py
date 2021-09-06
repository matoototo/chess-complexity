import json, argparse, pathlib

parser = argparse.ArgumentParser(description='Extract values for histogram generation.')
parser.add_argument('-i', '--input', metavar='path', type=pathlib.Path, help='the path to the input json file', required=True)
parser.add_argument('-o', '--output', metavar='path', type=pathlib.Path, help='the path to the output txt file', required=True)
parser.add_argument('-f', '--field', metavar='name', type=str, help='the field to extract', required=True)
parser.add_argument('--ignore', nargs='+', metavar='values', type=int, help='the values to ignore', required=False, default=[])

args = parser.parse_args()
pos = json.load(open(args.input, 'r+'))
values = list(map(lambda x : x[args.field], pos))
for_hist = open(args.output, 'w+')
for e in values:
    if e not in args.ignore: for_hist.write(f"{e}\n")
