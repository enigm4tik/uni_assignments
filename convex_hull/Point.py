class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def subtract(self, point):
    return Point(self.x - point.x, self.y - point.y)

  def __str__(self):
    return '(' + str(self.x) + ', ' + str(self.y) + ')'