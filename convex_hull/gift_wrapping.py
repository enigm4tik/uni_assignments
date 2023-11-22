from helper import direction
import matplotlib.pyplot as plt
import matplotlib as matlib

def gift_wrapping(points):
  """
  Calculate the convex hull using the Gift Wrapping Algorithm or
  Jarvis' March. 
  Find the left most point and then each point that is counter-clockwise.
  If the orientation is colinear calculate the farthest point. 
  :param points: list of Point objects
  :return: list of Point objects on Convex Hull
  """
  left_most_point = min(points, key=lambda point: point.x)
  index = points.index(left_most_point)

  left_index = index
  result = []
  result.append(left_most_point)
  while (True):
    next_in_list = (left_index + 1) % len(points)
    for i in range(len(points)):
      if i == left_index:
        continue
      orientation = direction(points[left_index], points[i],
                              points[next_in_list])
      if orientation > 0:
        next_in_list = i
    left_index = next_in_list
    if left_index == index:
      break
    result.append(points[next_in_list])
  return result
  

def gift_wrapping_animated(points, ax):
  list_of_points_x = [point.x for point in points]
  list_of_points_y = [point.y for point in points]
  plt.scatter(list_of_points_x, list_of_points_y, s=10)
  
  left_most_point = min(points, key=lambda point: point.x)
  
  index = points.index(left_most_point)

  left_index = index
  result = []
  result.append(left_most_point)
  while (True):
    next_in_list = (left_index + 1) % len(points)
    for i in range(len(points)):
      l = matlib.lines.Line2D([list_of_points_x[left_index], list_of_points_x[next_in_list]], 
                              [list_of_points_y[left_index], list_of_points_y[next_in_list]], ls='--')
      plt.waitforbuttonpress()
      ax.add_line(l)
      if i == left_index:
        l = matlib.lines.Line2D([list_of_points_x[left_index], list_of_points_x[next_in_list]], 
                              [list_of_points_y[left_index], list_of_points_y[next_in_list]], ls='--')
        plt.waitforbuttonpress()
        ax.add_line(l)
        continue
      orientation = direction(points[left_index], points[i],
                              points[next_in_list])
      if orientation > 0:
        next_in_list = i
        l = matlib.lines.Line2D([list_of_points_x[left_index], list_of_points_x[next_in_list]], 
                              [list_of_points_y[left_index], list_of_points_y[next_in_list]], ls='--')
        plt.waitforbuttonpress()
        ax.add_line(l)
    left_index = next_in_list
    l = matlib.lines.Line2D([list_of_points_x[left_index], list_of_points_x[next_in_list]], 
                              [list_of_points_y[left_index], list_of_points_y[next_in_list]], ls='--')
    plt.waitforbuttonpress()
    ax.add_line(l)
    if left_index == index:
      break
    result.append(points[next_in_list])
    l = matlib.lines.Line2D([result[-2].x, result[-1].x], [result[-2].y, result[-1].y], ls='-')
    plt.waitforbuttonpress()
    ax.add_line(l)
  l = matlib.lines.Line2D([result[-1].x, result[0].x], [result[-1].y, result[0].y], ls='-')
  plt.waitforbuttonpress()
  ax.add_line(l)
  return result