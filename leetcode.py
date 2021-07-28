import random
import string

from typing import List

generated_list = [random.randint(0, 10) for _ in range(0, random.randint(0, 100))]
generated_num = random.randint(1, random.randint(2, 10))
generated_string = ''.join(random.choice(string.ascii_uppercase) for _ in range(30))


def generated_list_digits_sorted(N):
    x = [random.randint(0, 100) for _ in range(N)]
    x.sort()
    return x


def printer(f):
    def wrapped(*args, **kwargs):
        r = f(*args, **kwargs)
        print(r)
        return r

    return wrapped


@printer
def twoSum(nums: List[int], target: int) -> List[int]:
    """
    1.
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
    You may assume that each input would have exactly one solution, and you may not use the same element twice.
    You can return the answer in any order.
    """
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
    """
    2.
    You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
    You may assume the two numbers do not contain any leading zero, except the number 0 itself.
    """
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
    """
    3.
    Given a string s, find the length of the longest substring without repeating characters.
    """
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


@printer
def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    """
    4.
    Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.
    The overall run time complexity should be O(log (m+n)).
    """
    unified_arr = []
    while nums1 or nums2:
        if not nums1:
            unified_arr += nums2
            break
        if not nums2:
            unified_arr += nums1
            break
        if nums1[0] < nums2[0]:
            unified_arr.append(nums1.pop(0))
        else:
            unified_arr.append(nums2.pop(0))
    l = len(unified_arr)
    print(unified_arr)
    if l % 2 == 1:
        return unified_arr[l // 2]
    return (unified_arr[(l - 1) // 2] + unified_arr[l // 2]) / 2


def longestPalindrome(s: str) -> str:
    """
    5.
    Given a string s, return the longest palindromic substring in s.
    """

    def expand(s, i, j):
        counter = j - i + 1
        i -= 1
        j += 1
        while i >= 0 and j < len(s):
            if s[i] == s[j]:
                counter += 2
                i -= 1
                j += 1
            else:
                return s[i + 1:j]
        return s[i + 1:j]

    if len(s) == 1:
        return s[0]
    max_length = ""
    for start in range(len(s) - 1):
        if s[start] == s[start + 1]:
            length = expand(s, start, start + 1)
            max_length = max_length if len(max_length) > len(length) else length
        length = expand(s, start, start)
        max_length = max_length if len(max_length) > len(length) else length
    return max_length


def convert(s: str, numRows: int) -> str:
    """
    6.
    The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
    P   A   H   N
    A P L S I I G
    Y   I   R
    And then read line by line: "PAHNAPLSIIGYIR"
    Write the code that will take a string and make this conversion given a number of rows
    """
    if numRows == 1 or len(s) <= numRows:
        return s
    row = 0
    up = True
    st = ''
    mat = {}
    for char in s:
        if row == numRows - 1:
            up = False
        elif row == 0:
            up = True
        if row not in mat:
            mat[row] = ''
        mat[row] += char
        if up:
            row += 1
        else:
            row -= 1
    for i in range(numRows):
        st += mat[i]
    return st


if __name__ == '__main__':
    findMedianSortedArrays([1, 2], [3, 4])
    lengthOfLongestSubstring("aabaab!bb")
    twoSum([10, 5, 10, 7, 5, 6, 6, 3, 4, 4, 8, 3, 4, 5, 6, 0], 3)
