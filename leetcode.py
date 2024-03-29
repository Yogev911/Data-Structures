import random
import string
from collections import defaultdict

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


@printer
def reverse(x: int) -> int:
    """
    7.
    Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
    Assume the environment does not allow you to store 64-bit integers (signed or unsigned).
    """
    sign = 1 if x > 0 else -1
    x *= sign
    powers = 0
    ret_val = 0
    while x // pow(10, powers):
        powers += 1
    for i in range(powers - 1, -1, -1):
        num = x % 10
        x //= 10
        ret_val += num * pow(10, i)
    ret_val *= sign
    if not (-1 * pow(2, 31) <= ret_val <= 1 * pow(2, 31) - 2):
        return 0
    return ret_val


@printer
def myAtoi(s: str) -> int:
    """
    8.
    Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).
    """
    if not s:
        return 0
    sign = 1
    num = 0
    start_index = 0
    for i in range(len(s)):
        if s[i] == ' ':
            continue
        if s[i] == '-' or s[i] == '+':
            sign = [1, -1][s[i] == '-']
            start_index = i + 1
            break
        start_index = i
        break

    for i in range(start_index, len(s)):
        if not s[i].isdigit():
            break
        if s[i] == '0' and num == 0:
            continue
        num = num * 10 + int(s[i])
    ret = num * sign
    if ret < -2 ** 31:
        return -2 ** 31
    if ret > 2 ** 31 - 1:
        return 2 ** 31 - 1
    return ret


@printer
def isPalindrome(x: int) -> bool:
    """
    9.
    Given an integer x, return true if x is palindrome integer.
    An integer is a palindrome when it reads the same backward as forward. For example, 121 is palindrome while 123 is not.
    """
    if x < 0:
        return False
    s = str(x)
    le = len(s)
    if le == 1:
        return True
    if le == 2:
        return s[0] == s[1]
    end = s[le // 2:] if le % 2 == 0 else s[le // 2 + 1:]
    return s[:le // 2][::-1] == end


def maxArea(height: List[int]) -> int:
    """
    11.
    Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai).
    n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i, 0).
    Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.
    """
    current = 0
    j = len(height) - 1
    i = 0
    while i != j:
        current = max(current, (j - i) * min(height[i], height[j]))
        if height[i] > height[j]:
            j -= 1
        else:
            i += 1
    return current


def detectCapitalUse(word: str) -> bool:
    """
    520. Detect Capital  - Easy
    We define the usage of capitals in a word to be right when one of the following cases holds:
    All letters in this word are capitals, like "USA".
    All letters in this word are not capitals, like "leetcode".
    Only the first letter in this word is capital, like "Google".
    Given a string word, return true if the usage of capitals in it is right.
    """
    if len(word) == 1:
        return True
    first_char = word[0].isupper()
    count_upper = int(first_char)
    for char in word[1:]:
        count_upper += int(char.isupper())
    if count_upper == len(word):
        return True
    if int(first_char) == count_upper:
        return True
    return False


@printer
def areSentencesSimilar(sentence1: str, sentence2: str) -> bool:
    """
    1813. Sentence Similarity III - Medium
    A sentence is a list of words that are separated by a single space with no leading or trailing spaces. For example, "Hello World", "HELLO", "hello world hello world" are all sentences. Words consist of only uppercase and lowercase English letters.
    Two sentences sentence1 and sentence2 are similar if it is possible to insert an arbitrary sentence (possibly empty) inside one of these sentences such that the two sentences become equal. For example, sentence1 = "Hello my name is Jane" and sentence2 = "Hello Jane" can be made equal by inserting "my name is" between "Hello" and "Jane" in sentence2.
    Given two sentences sentence1 and sentence2, return true if sentence1 and sentence2 are similar. Otherwise, return false.
    """
    sentence1 = sentence1.split(' ')
    sentence2 = sentence2.split(' ')
    if len(sentence1) < len(sentence2):
        sentence1, sentence2 = sentence2, sentence1

    while sentence2:
        if sentence1[-1] == sentence2[-1]:
            sentence1.pop()
            sentence2.pop()
        else:
            break
    sentence1 = sentence1[::-1]
    sentence2 = sentence2[::-1]
    while sentence2:
        if sentence1[-1] == sentence2[-1]:
            sentence1.pop()
            sentence2.pop()
        else:
            break
    return not bool(len(sentence2))


@printer
def binaryGap(n: int) -> int:
    """
    868. Binary Gap - Easy
    Given a positive integer n, find and return the longest distance between any two adjacent 1's in the binary representation of n. If there are no two adjacent 1's, return 0.
    Two 1's are adjacent if there are only 0's separating them (possibly no 0's). The distance between two 1's is the absolute difference between their bit positions. For example, the two 1's in "1001" have a distance of 3.
    """
    bin_num = bin(n)[2:]
    bin_num = [int(x) for x in bin_num]
    count = 0
    max_count = 0
    for num in bin_num[1:]:
        count += 1
        if num == 1:
            max_count = max(count, max_count)
            count = 0

    return max_count


def addDigits(num: int) -> int:
    """
    258. Add Digits - Easy
    Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.
    """
    while num > 9:
        digits = [int(x) for x in str(num)]
        num = sum(digits)
    return num


def isHappy(n: int) -> bool:
    """
    202. Happy Number - Easy
    Write an algorithm to determine if a number n is happy.
    A happy number is a number defined by the following process:
    Starting with any positive integer, replace the number by the sum of the squares of its digits.
    Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
    Those numbers for which this process ends in 1 are happy.
    Return true if n is a happy number, and false if not.
    """
    h_set = set()
    while n != 1:
        if n in h_set:
            return False
        digits = [int(x) ** 2 for x in str(n)]
        h_set.add(n)
        n = sum(digits)
    return True


if __name__ == '__main__':
    isHappy(9876)
    # binaryGap(15)
    # isPalindrome(1)
    # myAtoi("21474836460")
    # reverse(1234689)
    # findMedianSortedArrays([1, 2], [3, 4])
    # lengthOfLongestSubstring("aabaab!bb")
    # twoSum([10, 5, 10, 7, 5, 6, 6, 3, 4, 4, 8, 3, 4, 5, 6, 0], 3)
