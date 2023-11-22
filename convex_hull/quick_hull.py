from helper import direction, calculate_distance
import matplotlib.pyplot as plt
import matplotlib as matlib

def quick_hull(points):
    """
    Calculate the convex hull using the Quick Hull Algorithm.
    Find the left and right most point and connect them to separate the area in upper and lower. 
    Then for each half iteratively find a point on the hull. 
    :param points: list of Point objects
    :return: list of Point objects on Convex Hull
    """
    convex_hull = []
    
    left_most_point = min(points, key=lambda point: point.x)
    right_most_point = max(points, key=lambda point: point.x)
    
    convex_hull.append(left_most_point)
    convex_hull.append(right_most_point)
    
    left_side = []
    right_side = []

    for current_point in points:
      if (direction(left_most_point, right_most_point, current_point) < 0):
        left_side.append(current_point)
      if (direction(left_most_point, right_most_point, current_point) > 0):    
        right_side.append(current_point)
                
    find_hull(left_side, left_most_point, right_most_point, convex_hull)
    find_hull(right_side, right_most_point, left_most_point, convex_hull)
    return convex_hull
  

def find_hull(points, left_most_point, right_most_point, convex_hull):
    """
    Find the furthest point from and connect it to the two previous found points.
    Every point that is on the right side of the line (inside the triangle) is not on the hull.
    Continue with the remaining points using the new hull point until no points are left.
    :param points: list of Point objects
    :param left_most_point: Point object
    :param right_most_point: Point object
    :param convex_hull: list of Point objects on the convex hull to update
    :return: updated list of Point objects on Convex Hull
    """
    if(len(points) == 0): return convex_hull
   
    furthest_point = points[0]
    maxDist = 0
  
    for current_point in points:      
        dist = calculate_distance(left_most_point, right_most_point, current_point)
        if(dist > maxDist): 
            maxDist = dist
            furthestPoint = current_point

    points.remove(furthestPoint)
    convex_hull.append(furthestPoint)

    left_side = []
    right_side = []
    
    for point in points:
      if direction(left_most_point, furthestPoint, point) < 0:
        left_side.append(point)
      elif direction(furthestPoint, right_most_point, point) < 0:
        right_side.append(point)
        
    find_hull(left_side, left_most_point, furthestPoint, convex_hull)
    find_hull(right_side, furthestPoint, right_most_point, convex_hull)