# Convex Hull
from Point import Point
from gift_wrapping import gift_wrapping
from quick_hull import quick_hull
from timeit import default_timer as timer
from helper import generate_random_points, read_file, handle_random_arg, create_best_case_square, create_worst_case
import argparse, json

parser = argparse.ArgumentParser(prog="Convex Hull")
parser.add_argument("-f", "--file", help="Choose a text file, first line amount of numbers, comma separated float pairs on all other lines ")
parser.add_argument("-m", "--mode", help="Choose between 'interactive' and 'simple', default: simple")
parser.add_argument("-r", "--random", help="Generate random points: n max min [default: 100 1000 0]", nargs='+')
args = parser.parse_args()
list_of_points = []
if args.file and args.random:
  exit("Please choose between numbers from a file and random numbers.")

if args.file:
  list_of_points = read_file(args.file)
  for point in list_of_points:
      print(point)

if args.random:
  list_of_points = handle_random_arg(args.random)

if args.mode == 'interactive' or args.mode == 'i':
  # show visualisation
  fig, ax = plt.subplots(1, 1)
  ani = FuncAnimation(fig, gift_wrapping_animated(list_of_points, ax), interval=10000, save_count = 50, repeat=False)
  fig, ax = plt.subplots(1, 1)
  ani = FuncAnimation(fig, quick_hull_animated(list_of_points, ax), interval=10000, save_count = 50, repeat=False)
  plt.show()
else:
  if not list_of_points:
    list_of_points = generate_random_points(amount=10000, min=-5000, max=5000)
  print(f"Calculating the convex hull of {len(list_of_points)} points.")
  start = timer()
  convex_hull_by_quick_hull = quick_hull(list_of_points)
  end = timer()
  print(f"QuickHull: {round(end - start, 3)} seconds. {len(convex_hull_by_quick_hull)}")
  start = timer()
  convex_hull_by_gift_wrapping = gift_wrapping(list_of_points)
  end = timer()
  print(f"Gift Wrapping: {round(end - start, 3)} seconds. {len(convex_hull_by_gift_wrapping)}")

import math
pi = math.pi
