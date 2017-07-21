import bisect

def map(value):

	classes = 'ABC'
	class_threshold = (1, 2, 3)
	for i in value:
		B = bisect.bisect(class_threshold, value)
		new = value[B]

array = [1, 2, 3]
map(array)