from Point import Point
import random
from math import pi, cos, sin
from timeit import default_timer as timer


def cross_product(point1, point2):
  return point1.x * point2.y - point2.x * point1.y


def direction(point1, point2, point3):
  direction = cross_product(point3.subtract(point1), point2.subtract(point1))
  return direction


def generate_random_points(min=0, max=300, amount=50):
  list_of_points = []
  print(f"Calculating {amount} numbers between {min} and {max}. This may take a while.")
  start = timer()
  for i in range(amount):
    list_of_points.append(
      Point(random.uniform(min, max),
            random.uniform(min, max)))
  end = timer()
  print(f"Number generation: {round(end - start, 3)} seconds.")
  return list_of_points


def calculate_distance(point1, point2, point3):
  return abs(direction(point1, point2, point3))


def read_file(file):
  point_list = []
  with open(file, 'r') as f:
    for index, line in enumerate(f.readlines()):
      if index != 0:
        x, y = line.split(',')
        x, y = float(x), float(y)
        point_list.append(Point(x, y))
  return point_list


def handle_random_arg(argument_list):
  min = 0
  max = 1000
  n = 100
  argument_length = len(argument_list)
  if argument_length == 1:
    n = int(argument_list[0])
  if argument_length == 2:
    n, max = [int(value) for value in argument_list]
  if argument_length == 3:
    n, min, max = [int(value) for value in argument_list]
  if argument_length > 3:
    exit("Too many values provided, a max of 3 values is allowed: n max min")

  list_of_points = generate_random_points(amount=n, min=min, max=max)  
  return list_of_points


def create_best_case_triangle(point1, point2, point3, n=10000):
  point_list = []
  i = 0
  while len(point_list) < n:
    new_point = Point(random.uniform(0, 10000),
            random.uniform(0, 10000))
    if direction(point1, point3, new_point) > 0 and direction(point3, point2, new_point) >0 and direction(point2, point1, new_point) > 0:
      point_list.append(new_point)
    i += 1

  f = open("cases/best_case.txt", "a")
  f.write(f"{n}\n")
  for point in point_list:
    f.write(f"{point.x}, {point.y}\n")
  f.close()


def create_best_case_square(x0, x1, y0, y1, n=10000):
  point_list = [Point(x0, y0), Point(x0, y1), Point(x1, y0), Point(x1, y1)]
  while len(point_list) < n:
    new_point = Point(random.uniform(0, 10000),
            random.uniform(0, 10000))
    if (new_point.x < x1 and new_point.x > x0) and (new_point.y < y1 and new_point.y > y0):
      point_list.append(new_point)

  f = open("cases/best_case.txt", "w")
  f.write(f"{n}\n")
  for point in point_list:
    f.write(f"{point.x}, {point.y}\n")
  f.close()


def create_worst_case(r, n=10000):
  point_list = []
  for i in range(n):
    x = cos(pi * 2 * random.uniform(-1 * n, n)) * r
    y = sin(pi * 2 * random.uniform(-1 * n, n)) * r
    point_list.append(Point(x, y))
  f = open("cases/worst_case_quick_hull.txt", "w")
  f.write(f"{n}\n")
  for point in point_list:
    f.write(f"{point.x}, {point.y}\n")
  f.close()
