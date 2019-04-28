# def get_ps(a, target):
#     if not a:
#         return False
#     for num in a[::-1]:
#         if num <= target:
#             return num
#     return False
#
#
# def search(lst, target):
#     min = 0
#     max = len(lst) - 1
#     avg = int((min + max) / 2)
#     # uncomment next line for traces
#     # print(f"list : {lst} , min: {min}, max: {max}, avg: {avg}, target: {target}")
#     while (min < max):
#         if (lst[avg] == target):
#             return avg
#         elif (lst[avg] < target):
#             try:
#                 return avg + 1 + search(lst[avg + 1:], target)
#             except:
#                 return None
#         else:
#             return search(lst[:avg], target)
#     # avg may be a partial offset so no need to print it here
#     # print "The location of the number in the array is", avg
#     if not lst or lst[avg] >= target:
#         raise ValueError()
#     return avg
#
#
# people = [3,2,3,2,2]
#
# x = [1,2,4,6,7,9,7,54,3,2,45, 7,2,4,66,4,2,1,3,90,55]
# y= x+people
#
# limit = 6
# people.sort()
# counter = 0
# while people:
#     boat = people.pop()
#     if limit - boat:
#         num = search(people, limit - boat)
#         if num is not None:
#             people.pop(num)
#     counter += 1
#
#
# print(counter)
#
#
# nums1 = [1,2]
# nums2 = [3,4]
# nums1 += nums2
# le = len(nums1)
# if le == 1:
#     print(nums1[0])
# if le == 2:
#     print( (nums1[0]+nums1[1])/2)
# nums1.sort()
# if le % 2 != 0:
#     print(nums1[int(le/2)])
# print ((nums1[(le/2)-1] + nums1[(le/2)-2]) /2)
#
#
# def threeSum(nums):
#     d = {}
#     tmp_d = {}
#     for n in nums:
#         if n not in tmp_d:
#             tmp_d[n] = 1
#         else:
#             tmp_d[n] +=1
#     for k,v in tmp_d.items():
#         if v > 3:
#             for _ in range(0,v-3):
#                 nums.remove(k)
#     tmp_d = {}
#     for i, n in enumerate(nums):
#         if n not in tmp_d:
#             tmp_d[n] = i
#
#     for i1, a in enumerate(nums):
#         for i2, b in enumerate(nums):
#             neg_sum = (a + b) * -1
#             if i1 != i2 and neg_sum in tmp_d and i2 != tmp_d[neg_sum] and i1 != tmp_d[neg_sum]:
#                 tmp = [a, b, neg_sum]
#                 tmp.sort()
#                 tmp = tuple(tmp)
#                 _hash = hash(tmp)
#                 if _hash not in d:
#                     d[_hash] = list(tmp)
#     return list(d.values())
#
# x= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#
#
# print(threeSum(x))
#
# def reverse(x):
#     if len(bin(abs(x))[2:]) > 32:
#         return 0
#     a = []
#     is_neg = True if x < 0 else False
#     if x < 10 and not is_neg:
#         return x
#     x = abs(x)
#     while x > 0:
#         remain = x % 10
#         x = int(x / 10)
#         a.append(remain)
#     num = 0
#     index = 0
#     while a:
#         num += a.pop() * (10 ** index)
#         index += 1
#     return num if not is_neg else num*-1
#
# print(reverse(-21474834132))

# def is_polindrom(s):
#     if len(s) == 1:
#         return True
#     if len(s) == 0:
#         return True
#     if s[0] != s[-1]:
#         return False
#     else:
#         return is_polindrom(s[1:-1])
#
#
# def longestPalindrome(s):
#     if not s:
#         return ""
#     if len(s) == 1:
#         return s
#     subs = {}
#     curr_max_len = 0
#     for i1 in range(0, len(s)):
#         for i2 in range(len(s), 0, -1):
#             if s[i1] != s[i2-1]:
#                 continue
#             curr_pol = s[i1:i2]
#             curr_pol_len = len(curr_pol)
#             if curr_pol_len < curr_max_len:
#                 continue
#             if is_polindrom(curr_pol):
#                 return curr_pol
#             curr_max_len = curr_pol_len
#             if curr_pol_len not in subs:
#                 subs[curr_pol_len] = [curr_pol]
#             else:
#                 subs[curr_pol_len].append(curr_pol)
#     for i in range(1000,0,-1):
#         if i in subs:
#             candidates = subs[i]
#             for candidate in candidates:
#                 if is_polindrom(candidate):
#                     return candidate
#     return ""
#

# Python3 code to demonstrate Difference Array

# Creates a diff array D[] for A[] and returns
# it after filling initial values.
import random
import time


def initializeDiffArray(A):
    n = len(A)

    # We use one extra space because
    # update(l, r, x) updates D[r+1]
    D = [0 for i in range(0, n + 1)]

    D[0] = A[0]
    D[n] = 0

    for i in range(1, n):
        D[i] = A[i] - A[i - 1]
    return D


# Does range update
def update(D, l, r, x):
    D[l] += x
    D[r + 1] -= x


# Prints updated Array
def printArray(A, D):
    for i in range(0, len(A)):
        if (i == 0):
            A[i] = D[i]

        # Note that A[0] or D[0] decides
        # values of rest of the elements.
        else:
            A[i] = D[i] + A[i - 1]

        print(A[i], end=" ")

    print("")


def is_polindrom(s):
    if len(s) == 1:
        return True
    if len(s) == 0:
        return True
    if s[0] != s[-1]:
        return False
    else:
        return is_polindrom(s[1:-1])


def longestPalindrome(s):
    if not s:
        return ""
    if len(s) == 1:
        return s
    max_pol = 0, ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            if j - i > max_pol[0]:
                if is_polindrom(s[i:j]):
                    print(s[i:j])
                    max_pol = j - i, s[i:j]
    return max_pol[1]


def max_area(height):
    L = 0
    result = 0
    if not height: return 0
    R = len(height) - 1
    while L != R:
        result = max(result, min(height[L], height[R]) * (R - L))
        if height[L] < height[R]:
            L += 1
        else:
            R -= 1
    return result


# SetAll
class DS(object):
    def __init__(self):
        self.storage = {}
        self.allValue = (time.time(),)

    def get(self, idx):
        if idx not in self.storage:
            return None
        val = self.storage[idx]
        if val[0] < self.allValue[0]:
            return self.allValue[1]
        return val[1]

    def set(self, idx, val):
        self.storage[idx] = (time.time(), val)

    def setAll(self, val):
        self.allValue = (time.time(), val)


def print_x(n):
    for i in range(0, n):
        j = n - 1 - i
        for k in range(0, n):
            if (k == i or k == j):
                print("x", end="")
            else:
                print(" ", end="")
        print('\n', end='')
        # arr = []


def get_largest_delta(arr):
    if not arr:
        return 0
    max = 0
    result = 0,
    for _ in range(len(arr) - 1, -1, -1):
        if arr[_] > max:
            max = arr[_]
        tmp_result = max - arr[_]
        if tmp_result > result[0]:
            result = tmp_result, arr[_], max
    return result


def lotto(nums, k, candidates):
    if k == 0 or not nums:
        return candidates.keys()
    random.shuffle(nums)
    num = nums[-1]
    if num not in candidates:
        candidates[num] = None
        nums.pop()
        return lotto(nums, k - 1, candidates)
    else:
        return lotto(nums, k, candidates)


def fib_arr(x):
    '''
    time O(N)
    space O(N)
    '''
    fibo_arr = [0, 1]
    for i in range(x-1):
        fibo_arr.append(fibo_arr[-1] + fibo_arr[-2])
    print(fibo_arr[-1])


def fibo_loop(n):
    '''
    time O(N)
    space O(2)
    '''
    a, b = 1, 1
    for i in range(n - 1):
        a, b = b, a + b
    return a

def fibo(n):
    '''
    time O(2^N)
    space O(N)
    '''
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)
if __name__ == '__main__':
    n = 88
    fib_arr(n)
    print(fibo_loop(n))
    print(fibo(n))
    # print_x(n)
    # print(get_largest_delta([5, 99999, 23, 6, 8, 1, 7, 6, 4, 99, 9, 1, 0, 9, 8]))
    # print(lotto(list(set([1, 1, 1, 2, 1, 5, 7, 8])), 3, {}))
    #
    #
    # def parens(left, right, string):
    #     if left == 0 and right == 0:
    #         arr.append(string)
    #     if left > 0:
    #         parens(left - 1, right + 1, string + "(")
    #     if right > 0:
    #         parens(left, right - 1, string + ")")
    # parens(n, 0, "")
    # print(arr)
    # a = DS()
    # for _ in range(10):
    #     a.set(random.randint(0,200),random.randint(0,200))
    # a.set(2,200)
    # a.get(2)
    # a.set(2,400)
    # a.get(2)
    # a.setAll(30)
    # a.get(2)
    # Driver Code
    # A = [10, 5, 20, 40]
    #
    # # Create and fill difference Array
    # D = initializeDiffArray(A)
    #
    # # After below update(l, r, x), the
    # # elements should become 20, 15, 20, 40
    # update(D, 0, 1, 10)
    # printArray(A, D)
    #
    # # After below updates, the
    # # array should become 30, 35, 70, 60
    # update(D, 1, 3, 20)
    # update(D, 2, 2, 30)
    # printArray(A, D)
    # print(max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]))
