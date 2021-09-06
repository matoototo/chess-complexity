import json, argparse, pathlib

parser = argparse.ArgumentParser(description='Calculate the delta of the elo attributes between two given files.')
parser.add_argument('--inp1', metavar='path', type=pathlib.Path, help='the path to the first input json file', required=True)
parser.add_argument('--inp2', metavar='path', type=pathlib.Path, help='the path to the second input json file', required=True)
parser.add_argument('-o', '--output', metavar='path', type=pathlib.Path, help='the path to the output json file', required=True)

args = parser.parse_args()

pos1 = json.load(open(args.inp1, 'r+'))
pos2 = json.load(open(args.inp2, 'r+'))

result = []
for pos in pos2:
    found = list(filter(lambda x : x['fen'] == pos['fen'], pos1))
    if len(found) == 0: continue
    found = found[0]
    result.append({'fen': pos['fen'], 'elo2': pos['elo'], 'elo1': found['elo'], 'delta': pos['elo']-found['elo']})

result.sort(key = lambda x : x['delta'])
json.dump(result, open(args.output, 'w+'), indent=4)
