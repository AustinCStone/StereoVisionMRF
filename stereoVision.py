import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functools as ft

RIGHT = 'right'
LEFT = 'left'
UP = 'up'
DOWN = 'down'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def load_images(left_dir, right_dir):
	left_img = rgb2gray(mpimg.imread(left_dir))
	right_img = rgb2gray(mpimg.imread(right_dir))
	return left_img, right_img


def produce_disparity_img(left_dir='tsukuba-imL.png', right_dir='tsukuba-imR.png',\
	max_disparity=16, message_passing_rounds=5, block_size=5):
	left_img, right_img = load_images(left_dir, right_dir)
	height, length = left_img.shape
	disparity_map = np.zeros((height, length, max_disparity))
	for h in range(height):
		for l in range(length):
			disparity_map[h, l] = get_cost_by_disparity(left_img, right_img, (h,l), max_disparity, block_size)
	message_map = pass_messages(disparity_map, message_passing_rounds)
	final_disparity_map = get_belief(disparity_map, message_map)
	plt.imshow(final_disparity_map, cmap = plt.get_cmap('gray'))
	plt.show()


# basically I just guessed on these values. using a truncated linear smoothness penalty
def get_smoothness_penalty(self_disparity, neighbors_disparity, penalty_constant=.05, cutoff=3.):
	return penalty_constant * min(abs(self_disparity - neighbors_disparity), cutoff)


# there is probably a standard better/clearer way to implement this
def pass_messages(disparity_map, message_passing_rounds):
	directions = [RIGHT, LEFT, UP, DOWN]
	height, length, max_disparity = disparity_map.shape
	message_map = np.ones((height, length, len(directions), max_disparity))
	next_message_map = np.ones((height, length, len(directions), max_disparity))
	for m_round in range(message_passing_rounds):
		for direction in directions:
			for h in range(height):
				for l in range(length):
					neighbor_messages = []
					# exclude the message from the neighbor we are passing to
					for i in range(len(directions)):
						if directions[i] != direction:
							neighbor_messages.append(message_map[h, l, i])
					# populate the new message map with the next round's messages
					try:
						if direction == RIGHT:
								next_message_map[h, l+1, directions.index(LEFT)] = \
									max_product(disparity_map[h, l], neighbor_messages)
						if direction == LEFT:
								next_message_map[h, l-1, directions.index(RIGHT)] = \
									max_product(disparity_map[h, l], neighbor_messages)
						if direction == UP:
								next_message_map[h+1, l, directions.index(DOWN)] = \
									max_product(disparity_map[h, l], neighbor_messages)
						if direction == DOWN:
								next_message_map[h-1, l, directions.index(UP)] = \
									max_product(disparity_map[h, l], neighbor_messages)
					except:
						pass
				print h
				print m_round
		message_map = next_message_map
	return message_map


def get_belief(disparity_map, message_map):
	height, length, max_disparity = disparity_map.shape
	final_disparity_map = np.zeros((height, length))
	for h in range(height):
		for l in range(length):
			best_disparity = 0
			best_disparity_cost = sys.maxint
			for disparity in range(max_disparity):
				neighbor_cost = 0.
				for neighbor in message_map[h, l]:
					neighbor_cost += neighbor[disparity]
				if disparity_map[h, l, disparity] + neighbor_cost < best_disparity_cost:
					best_disparity_cost = disparity_map[h, l, disparity] + neighbor_cost
					best_disparity = disparity
			final_disparity_map[h, l] = best_disparity / float(max_disparity)
	return final_disparity_map


def max_product(self_disparity_cost, neighbor_messages):
	max_disparity = len(self_disparity_cost)
	normalization_constant = 0.
	message_vec = [sys.maxint for i in range(max_disparity)]
	for disparity1 in range(max_disparity): # message recipient disparity
		for disparity2 in range(max_disparity): # own disparity
			neighbor_message_contribution = 0.
			for neighbor_message in neighbor_messages:
				neighbor_message_contribution += neighbor_message[disparity2]
			possible_val = self_disparity_cost[disparity2] + get_smoothness_penalty(\
				disparity1, disparity2) + neighbor_message_contribution
			if possible_val < message_vec[disparity1]:
				message_vec[disparity1] = possible_val
	message_vec = message_vec / np.sum(message_vec)
	return message_vec


def get_block_pixel_diff(left_img, right_img, left_location, disparity, block_size):
	left_block = left_img[left_location[0]-block_size/2:left_location[0]+block_size/2, \
		left_location[1]-block_size/2:left_location[1]+block_size/2]
	right_location = (left_location[0], left_location[1] - disparity) # images are already row aligned
	right_block = right_img[right_location[0]-block_size/2:right_location[0]+block_size/2, \
		right_location[1]-block_size/2:right_location[1]+block_size/2]
	return np.sum((right_block - left_block)**2.)


def get_cost_by_disparity(left_img, right_img, left_location, max_disparity, block_size):
	disparity_cost = [0. for i in range(max_disparity)]
	max_shift = max_disparity + block_size / 2
	if left_location[0] - max_shift < 0 or left_location[1] - max_shift < 0 or \
		left_location[0] + max_shift > len(left_img) or left_location[1] + max_shift > len(left_img[0]):
		return disparity_cost
	disparity = 0
	while disparity < max_disparity:
		disparity_cost[disparity] = get_block_pixel_diff(left_img, right_img, left_location, disparity, block_size)
		disparity += 1
	return disparity_cost


produce_disparity_img()