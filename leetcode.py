import random

from typing import List


def printer(f):
    def wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        print(r)
        return r

    return wrapped


@printer
def twoSum(nums: List[int], target: int) -> List[int]:
    '''
    1.
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.
    '''
    h_map = {}
    for i in range(len(nums)):
        candidate = target - nums[i]
        if candidate not in h_map:
            h_map[nums[i]] = i
        else:
            return [h_map[candidate], i]


if __name__ == '__main__':
    generated_list = [random.randint(0, 10) for _ in range(0, random.randint(0, 100))]
    generated_num = random.randint(1, random.randint(2, 10))
    twoSum([10, 5, 10, 7, 5, 6, 6, 3, 4, 4, 8, 3, 4, 5, 6, 0], 3)
