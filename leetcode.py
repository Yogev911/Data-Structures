import random
import string

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


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


@printer
def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    '''
    2.
    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    '''
    head = current = ListNode()
    val = 0
    while l1 or l2:
        if l1:
            val += l1.val
            l1 = l1.next
        if l2:
            val += l2.val
            l2 = l2.next

        current.next = ListNode(val=val % 10)
        current = current.next

        val //= 10
    if val:
        current.next = ListNode(val=val)
    return head


@printer
def lengthOfLongestSubstring(s: str) -> int:
    '''
    Given a string s, find the length of the longest substring without repeating characters.
    '''
    max_subset = 0
    h_map = {}
    start_index = 0
    for end_index in range(len(s)):
        if s[end_index] not in h_map:
            h_map[s[end_index]] = end_index
        else:
            if h_map[s[end_index]] < start_index:
                h_map[s[end_index]] = end_index
            else:
                start_index = h_map[s[end_index]] + 1
                h_map[s[end_index]] = end_index
        max_subset = max(max_subset, end_index - start_index + 1)
    return max_subset


if __name__ == '__main__':
    generated_list = [random.randint(0, 10) for _ in range(0, random.randint(0, 100))]
    generated_num = random.randint(1, random.randint(2, 10))
    generated_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(30))
    twoSum([10, 5, 10, 7, 5, 6, 6, 3, 4, 4, 8, 3, 4, 5, 6, 0], 3)
    lengthOfLongestSubstring("aabaab!bb")
