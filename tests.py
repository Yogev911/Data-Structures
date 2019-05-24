import datetime
import math
import random
import string
import time
from functools import wraps

from utils import timer


def search(lst, target):
    min = 0
    max = len(lst) - 1
    avg = int((min + max) / 2)
    while (min < max):
        if (lst[avg] == target):
            return avg
        elif (lst[avg] < target):
            try:
                return avg + 1 + search(lst[avg + 1:], target)
            except:
                return None
        else:
            return search(lst[:avg], target)
    # avg may be a partial offset so no need to print it here
    # print "The location of the number in the array is", avg
    if not lst or lst[avg] >= target:
        raise ValueError()
    return avg


def max_boats():
    # i have no idea what is it..
    people = [3, 2, 3, 2, 2]
    limit = 6
    people.sort()
    counter = 0
    while people:
        boat = people.pop()
        if limit - boat:
            num = search(people, limit - boat)
            if num is not None:
                people.pop(num)
        counter += 1

    print(counter)

    nums1 = [1, 2]
    nums2 = [3, 4]
    nums1 += nums2
    le = len(nums1)
    if le == 1:
        print(nums1[0])
    if le == 2:
        print((nums1[0] + nums1[1]) / 2)
    nums1.sort()
    if le % 2 != 0:
        print(nums1[int(le / 2)])
    print((nums1[(le / 2) - 1] + nums1[(le / 2) - 2]) / 2)


def three_sum(nums):
    '''
    input [1, 2, 4, 6, 7, 9, 7, 54, 3, 2, 45, 7, 2, 4, 66, 4, 2, 1, 3, 90, 55]
    :param nums:
    :return: [i1,i2,i3]
    '''
    d = {}
    tmp_d = {}
    for n in nums:
        if n not in tmp_d:
            tmp_d[n] = 1
        else:
            tmp_d[n] += 1
    for k, v in tmp_d.items():
        if v > 3:
            for _ in range(0, v - 3):
                nums.remove(k)
    tmp_d = {}
    for i, n in enumerate(nums):
        if n not in tmp_d:
            tmp_d[n] = i

    for i1, a in enumerate(nums):
        for i2, b in enumerate(nums):
            neg_sum = (a + b) * -1
            if i1 != i2 and neg_sum in tmp_d and i2 != tmp_d[neg_sum] and i1 != tmp_d[neg_sum]:
                tmp = [a, b, neg_sum]
                tmp.sort()
                tmp = tuple(tmp)
                _hash = hash(tmp)
                if _hash not in d:
                    d[_hash] = list(tmp)
    return list(d.values())


def reverse(x):
    '''
    reverse number ex . reverse(-21474834132)
    :param x:
    :return:
    '''
    if len(bin(abs(x))[2:]) > 32:
        return 0
    a = []
    is_neg = True if x < 0 else False
    if x < 10 and not is_neg:
        return x
    x = abs(x)
    while x > 0:
        remain = x % 10
        x = int(x / 10)
        a.append(remain)
    num = 0
    index = 0
    while a:
        num += a.pop() * (10 ** index)
        index += 1
    return num if not is_neg else num * -1


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
    subs = {}
    curr_max_len = 0
    for i1 in range(0, len(s)):
        for i2 in range(len(s), 0, -1):
            if s[i1] != s[i2 - 1]:
                continue
            curr_pol = s[i1:i2]
            curr_pol_len = len(curr_pol)
            if curr_pol_len < curr_max_len:
                continue
            if is_polindrom(curr_pol):
                return curr_pol
            curr_max_len = curr_pol_len
            if curr_pol_len not in subs:
                subs[curr_pol_len] = [curr_pol]
            else:
                subs[curr_pol_len].append(curr_pol)
    for i in range(1000, 0, -1):
        if i in subs:
            candidates = subs[i]
            for candidate in candidates:
                if is_polindrom(candidate):
                    return candidate
    return ""


def longest_palindrome2(s):
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


def longestPalindrome3(s):
    def is_polindrom(s):
        left = 0
        right = len(s) - 1
        while right - left > 0:
            if s[right] != s[left]:
                return False
            left += 1
            right -= 1
        return True

    if not s:
        return ""
    if len(s) == 1:
        return s
    max_pol = 0, ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            if j - i >= max_pol[0]:
                if is_polindrom(s[i:j + 1]):
                    max_pol = j - i, s[i:j + 1]
                return max_pol[1]


# UPDATE ALL IN O(1)
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


#


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


def print_x(n):
    '''
    in size n
    X X
     X
    X X
    :param n:
    :return:
    '''
    for i in range(0, n):
        k = n - 1 - i
        for j in range(0, n):
            if (j == i or j == k):
                print("x", end="")
            else:
                print(" ", end="")
        print('\n', end='')


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


def create_arr_size_k_of_n(k, n):
    t = 1
    arr = []
    for _ in range(n):
        arr.append([0] * 5)
    for i in range(n):
        for j in range(n):
            arr[i][j] = t % k + 1
            t += 1
    print(arr)


@timer
def lotto(nums, k, candidates):
    if k == 0 or not nums:
        return candidates.keys()
    random.shuffle(nums)
    num = nums.pop()
    if num not in candidates:
        candidates[num] = None
        return lotto(nums, k - 1, candidates)
    else:
        print(k)
        return lotto(nums, k, candidates)


@timer
def lotto2(nums, k):
    candidates = {}
    random.shuffle(nums)
    while k != 0 and nums:
        num = nums.pop()
        if num not in candidates:
            k -= 1
            candidates[num] = None
    return list(candidates.keys())


def fib_arr(x):
    '''
    time O(N)
    space O(N)
    '''
    fibo_arr = [0, 1]
    for i in range(x - 1):
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


import shutil


def print_fib(lb, ub, n):
    if n == 0:
        print("")
        return
    else:
        try:
            print(str(ub) + ",", end="")
            print_fib(ub, lb + ub, n - 1)
        except:
            print(f"\n max {ub}")
            return


def isMatch2(s, p):
    """
    :type s: str
    :type p: str
    :rtype: bool
    """
    n = len(s)
    m = len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(m):
        for j in range(-1, n):
            if j == -1:
                if p[i] == "*":
                    dp[i + 1][j + 1] = dp[i - 1][j + 1]
                continue
            if p[i].isalpha():
                if p[i] == s[j]:
                    dp[i + 1][j + 1] = dp[i][j]
            elif p[i] == ".":
                dp[i + 1][j + 1] = dp[i][j]
            else:
                dp[i + 1][j + 1] = dp[i - 1][j + 1] or (dp[i + 1][j] and (p[i - 1] == s[j] or p[i - 1] == "."))
    return dp[-1][-1]


def isMatch(s, p, char=False):
    '''
    regex
    :param s:
    :param p:
    :param char:
    :return:
    '''
    if not s:
        if not p:
            return True
        if len(p) > 1 and p[1] == '*':
            return isMatch(s, p[2:])
        if p[0] == '.':
            return isMatch(s, p[1:])
        elif len(p) == 1 and char and p[0] == char:
            return True
        else:
            return False
    if not p:
        return False
    if len(p) > 1:
        if p[1] == '*' and p[0] == s[0]:
            if len(s) == 1 and len(p) > 2:
                return isMatch(s, p[2:])
            return isMatch(s[1:], p, s[0])
        elif p[1] == '*' and p[0] == '.':
            if len(p) > 2 and len(s) > 1 and s[1] == p[2]:
                return isMatch(s[1:], p[2:], s[0])
            else:
                return isMatch(s[1:], p, s[0])
        elif p[1] == '*' and (p[0] != '.' or [0] != s[0]):
            return isMatch(s, p[2:])
        elif s[0] == p[0]:
            return isMatch(s[1:], p[1:])
        elif s[0] == p[0] or p[0] == '.':
            return isMatch(s[1:], p[1:])
        else:
            return False
    elif s[0] == p[0] or p[0] == '.':
        return isMatch(s[1:], p[1:])
    else:
        return False


# AutoComplete
class ACnode:
    def __init__(self, prefix):
        self.prefix = prefix
        self.is_word = None
        self.childrens = {}


class AutoComplete:
    def __init__(self, words):
        self.root = ACnode('')
        self.root.is_word = False
        self._insert_words(words)

    def _insert_words(self, words):
        for word in words:
            self._insert_word(word)

    def _insert_word(self, word):
        curr = self.root
        for i, char in enumerate(word):
            if char not in curr.childrens:
                curr.childrens[char] = ACnode(word[:i + 1])
            curr = curr.childrens[char]
        curr.is_word = True

    def get_word_prefix(self, prefix):
        results = []
        curr = self.root
        for char in prefix:
            if char not in curr.childrens:
                return results
            curr = curr.childrens[char]

        self._get_all_childs(curr, results)
        return results

    def _get_all_childs(self, curr, results):
        if curr.is_word:
            results.append(curr.prefix)
        else:
            for char, node in curr.childrens.items():
                self._get_all_childs(node, results)


#

def list_factors(number):
    """Alternate list_factors implementation."""
    factors = []
    counter = 1
    while counter <= number:
        if number % counter == 0:
            factors.append(counter)
        counter += 1
    return factors


def is_prime(number):
    """Return true if the number is a prime, else false."""
    return len(list_factors(number)) == 2


def next_prime(number):
    """Return the next prime."""
    next_number = number + 1
    while not is_prime(next_number):
        next_number += 1
    return next_number


def prime_factorization(n):
    results = []
    prime = 2
    while n != 1:
        if n % prime == 0:
            results.append(prime)
            n = int(n / prime)
            prime = 2
        else:
            prime = next_prime(prime)
    return results


def prime_factorization_rec(n, prime=2):
    if n == 1:
        return
    if n % prime == 0:
        print(prime)
        prime_factorization_rec(int(n / prime))
    else:
        prime_factorization_rec(n, next_prime(prime))


# SHUFFLE SONGS
def shuffle_songs(songs_arr):
    '''
    :param songs_arr: lists of tuples -> (artist,song)
    :return: lists of shuffled tuples -> (artist,song)
    space O(n)
    time O(n*groups)
    '''
    groups = {}
    for song in songs_arr:
        if song[0] not in groups:
            groups[song[0]] = [song]
        else:
            groups[song[0]].append(song)
    max_number_of_songs = get_max_songs_by_artist(groups)
    unshuffled_songs = fill_random_nones_in_artist(groups, max_number_of_songs)  # 2d array
    shuffled_songs = []
    for i in range(max_number_of_songs):
        bulk = list(
            filter(lambda x: x is not None, map(lambda group: group[i] if group[i] else None, unshuffled_songs)))
        shuffled_songs += random.sample(bulk, len(bulk))
    return shuffled_songs


def get_max_songs_by_artist(groups):
    max = 0
    for k, v in groups.items():
        if len(v) > max:
            max = len(v)
    return max


def fill_random_nones_in_artist(groups, max_songs):
    songs = []
    for k, v in groups.items():
        if len(v) < max_songs:
            v += [None] * max_songs
            random.shuffle(v)
        songs.append(v)
    return songs


#

def letterCombinations(digits):
    key_map = {
        '2': ['a', 'b', 'c'],
        '3': ['d', 'e', 'f'],
        '4': ['g', 'h', 'i'],
        '5': ['j', 'k', 'l'],
        '6': ['m', 'n', 'o'],
        '7': ['p', 'q', 'r', 's'],
        '8': ['t', 'u', 'v'],
        '9': ['w', 'x', 'y', 'z']
    }

    class ACnode:
        def __init__(self, prefix):
            self.prefix = prefix
            self.done = None
            self.childrens = {}

    outputs = []
    root = ACnode("")
    curr = root
    words = [x for x in key_map[digits[0]]]
    for num in digits[1:]:
        words = [x + y for x in words for y in key_map[num]]
    print(words)


def max_subs(s):
    d = {}
    max_sub = 0
    start_ptr = 0
    for i in range(0, len(s)):
        if s[i] in d:
            window = i - start_ptr
            if d[s[i]] < start_ptr:
                window += 1
            if d[s[i]] + 1 > start_ptr:
                start_ptr = d[s[i]] + 1
        else:
            window = i - start_ptr + 1
        d[s[i]] = i
        if window > max_sub:
            max_sub = window
    return max_sub


def reverse_int(x):
    neg = [-1, 1][x >= 0]
    x = abs(x)
    num = 0
    while x != 0:
        mod = x % 10
        x = int(x / 10)
        num = num * 10 + mod
    num *= neg
    return num if -2 ** 31 < num and num < 2 ** 31 else 0


def divide(dividend, divisor):
    counter = 0
    sign = 1
    if dividend < 0:
        sign *= -1
    if divisor < 0:
        sign *= -1
    divisor = abs(divisor)
    dividend = abs(dividend)
    while dividend >= divisor:
        dividend -= divisor
        counter += 1
    num = counter * sign
    return num if num > -2 ** 31 and num < 2 ** 31 - 1 else 0


def removeDuplicates(nums):
    if not nums:
        return None
    next_num_index = 1
    max_num = max(nums)
    for i in range(1, len(nums)):
        if nums[i] > nums[next_num_index - 1]:
            nums[next_num_index] = nums[i]
            next_num_index += 1
    return len(nums[:next_num_index])


def match_strings_k_distinct(s1, s2, k):
    max_len = max(len(s1), len(s2))
    for i in range(max_len):
        try:
            if s1[i].upper() != s2[i].upper():
                if not k:
                    return False
                k -= 1
        except IndexError:
            if not k:
                return False
            k -= 1
    return True if not k else False


def subsets(nums, k):
    def get_subs(nums, i, subset, k):
        if i == len(nums):
            tmp_sub = [x for x in subset if x is not None]
            if tmp_sub and max(tmp_sub) + min(tmp_sub) <= k:
                return 1
            else:
                return 0
        else:
            subset[i] = None
            acc = get_subs(nums, i + 1, subset, k)
            subset[i] = nums[i]
            acc += get_subs(nums, i + 1, subset, k)
        return acc

    subset = [None] * len(nums)
    num = get_subs(nums, 0, subset, k)
    return num


def subsets2(nums, k):
    nums.sort()
    counter = 0
    for j in range(len(nums) - 1, -1, -1):
        i = 0
        for index in range(i, j):
            if nums[i] + nums[j] <= k:
                counter += 2 ** (j - i - 1)
    return counter


def numDecodings(s):
    def get_decoded(s, i):
        if i == 0:
            return 1
        start = len(s) - i
        if s[start] == '0':
            return 0
        result = get_decoded(s, i - 1)
        if i >= 2 and int(s[start:start + 2]) <= 26:
            result += get_decoded(s, i - 2)
        return result

    decodes = get_decoded(s, len(s))
    return decodes


def maxProfit(prices):
    max_val = -1
    max_price = -1
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] > max_price:
            max_price = prices[i]
        delta = max_price - prices[i - 1]
        if delta > max_val:
            max_val = delta
    return max_val


def moveZeroes(nums):
    """
    Do not return anything, modify nums in-place instead.
    """
    index = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[index] = nums[i]
            index += 1
    for i in range(len(nums) - index):
        nums[index + i] = 0
    print(nums)


def meeting_order(meetings):
    start = meetings[0][0]
    end = meetings[0][0]
    for meeting in meetings:
        if start > meeting[0]:
            start = meeting[0]
        if end < meeting[1]:
            end = meeting[1]
    span = [0] * (end - start + 1)
    for meeting in meetings:
        span[meeting[0] - start] = 1
        span[meeting[1] - start] = -1
    count = 0
    for cell in span:
        if count > 1:
            return False
        count += cell
    return True


def isPalindrome(s):
    def is_palindrome(s, start, end):
        if end - start + 1 == 1:
            return True
        if end - start + 1 == 0:
            return True
        if s[start] != s[end]:
            return False
        else:
            return is_palindrome(s, start + 1, end - 1)

    s = ''.join([ch.lower() for ch in s if ch in string.ascii_letters + string.digits])
    print(s)
    if not s:
        return True
    return is_palindrome(s, 0, len(s) - 1)


def isIntPalindrome(x):
    if x < 0:
        return False
    x = abs(x)
    num = 0
    tmp = x
    while tmp >= 1:
        num = num * 10 + (tmp % 10)
        tmp //= 10
    print(x, num)
    return x == num


def intToRoman(num):
    chars = {1: 'I', 5: 'V', 10: 'X', 50: 'L', 100: 'C', 500: 'D', 1000: 'M'}
    chars2 = {4: 'IV', 9: 'IX', 40: 'XL', 90: 'XC', 400: 'CD', 900: 'CM'}
    roman = []
    roman_keys = [1000, 500, 100, 50, 10, 5, 1]
    power = [1000, 100, 10, 1]
    pow_i = 0
    for pow_i, p in enumerate(power):
        if num // p != 0:
            break

    for i, key in enumerate(roman_keys):
        while num // key:
            while num < power[pow_i]:
                pow_i += 1
            tmp = (num // power[pow_i]) * power[pow_i]
            if tmp in chars2:
                roman.append(chars2[tmp])
                num -= tmp
            else:
                num -= key
                roman.append(chars[key])
    return ''.join(roman)


def maxArea(height):
    def get_area(p1, p2):
        return abs(p2[0] - p1[0]) * min(p2[1], p1[1])

    left = 0
    right = len(height) - 1
    max_num = 0
    while left != right:
        curr_area = get_area((left, height[left]), ((right, height[right])))
        if curr_area > max_num:
            max_num = curr_area
        if height[left] >= height[right]:
            right -= 1
        else:
            left += 1

    return max_num


def threeSum(nums):
    def get_nums(res):
        return [int(x) for x in res.split('.')]

    d = {}
    results = {}
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j:
                if -1 * (nums[i] + nums[j]) in d and d[-1 * (nums[i] + nums[j])] != i and d[
                    -1 * (nums[i] + nums[j])] != j:
                    curr_result = [nums[d[-1 * (nums[i] + nums[j])]], nums[i], nums[j]]
                    curr_result.sort()
                    curr_result = '.'.join(map(str, curr_result))
                    if curr_result not in results:
                        results[curr_result] = None
                    d[nums[j]] = j
                else:
                    d[nums[j]] = j
    return [get_nums(res) for res in list(results.keys())]


def maxAreaOfIsland(grid):
    def get_area(grid, i, j):
        if len(grid) > i and i >= 0 and len(grid[i]) > j and j >= 0 and grid[i][j] == 1:
            grid[i][j] = 0
            return 1 + get_area(grid, i + 1, j) + get_area(grid, i, j + 1) + get_area(grid, i - 1, j) + get_area(grid,
                                                                                                                 i,
                                                                                                                 j - 1)
        else:
            return 0

    max_area = 0
    cul_size = range(len(grid[0]))
    for i in range(len(grid)):
        for j in cul_size:
            max_area = max(max_area, get_area(grid, i, j))
    return max_area


def is_jump_valid(last_jump, from_index, to_index, stones):
    return stones[to_index] - stones[from_index] == last_jump or stones[to_index] - stones[
        from_index] == last_jump + 1 or \
           stones[to_index] - stones[from_index] == last_jump - 1


def sec_biggest_elem(arr):
    biggest = -2 ** 31
    sec_biggest = -2 ** 31
    for num in arr:
        if num > biggest:
            sec_biggest = biggest
            biggest = num
        elif sec_biggest > num:
            sec_biggest = num

    return sec_biggest


def test(a, *b, **c):
    try:
        print(a)
        raise
    except:
        print(b)
    finally:
        print(c)


from functools import wraps


def prefix(prefix):
    def my_deco(f):
        # @wraps(f)
        def wrapper(*args, **kwargs):
            elem = kwargs.get('bla', ' nothing.. ')
            return prefix + elem + f(*args, **kwargs)

        return wrapper

    return my_deco


@prefix("hello")
def testing(name):
    return f"hi {name}"


import heapq


def canCross(stones):
    for i in range(3, len(stones)):
        if stones[i] > stones[i - 1] * 2:
            return False
    last_stone = stones[-1]
    positions = []
    jumps = []
    valid_stones = {}
    for stone in stones:
        valid_stones[stone] = None
    positions.append(0)
    jumps.append(0)
    while positions:
        position = positions.pop()
        jump = jumps.pop()
        for j in range(jump - 1, jump + 2):
            if j <= 0:
                continue
            new_position = position + j
            if new_position == last_stone:
                return True
            elif new_position in valid_stones:
                positions.append(new_position)
                jumps.append(j)
    return False


def findPeakElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    left = 0
    right = len(nums) - 1
    while left < right:

        mid = left + int((right - left) / 2)
        print(left, mid, right)
        if nums[mid] < nums[mid + 1]:
            left = mid + 1
        else:
            right = mid
    return left


def backspaceCompare(S, T):
    """
    :type S: str
    :type T: str
    :rtype: bool
    """
    i = len(S) - 1
    j = len(T) - 1
    counter = {'a': [], 'b': []}
    while not (i <= 0 and j <= 0):
        while S and not S[i].isalpha():
            counter['a'].append(1)
            S = S[:i]
            i -= 1
        while S and S[i].isalpha() and counter['a']:
            counter['a'].pop()
            S = S[:i]
            i -= 1
        while T and not T[j].isalpha():
            counter['b'].append(1)
            T = T[:j]
            j -= 1
        while T and T[j].isalpha() and counter['b']:
            counter['b'].pop()
            T = T[:j]
            j -= 1

        if i < -1 or j < -1:
            return False
        if not T and not S:
            return True

        if (S and not S[i].isalpha()) or (T and not T[j].isalpha()):
            continue
        if S[-1] != T[-1]:
            return False
        i -= 1
        j -= 1
    return True


def threeSum2(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    d = {}
    results = {}
    for i in range(len(nums)):
        if nums[i] not in d:
            d[nums[i]] = [i]
        else:
            d[nums[i]].append(i)
    for i in range(len(nums)):
        for j in range(len(nums)):
            neg_num = (nums[i] + nums[j]) * -1
            if i != j and neg_num in d:
                for index in d[neg_num]:
                    if i != index and j != index:
                        candidate = hash(tuple((sorted([nums[i], nums[j], nums[index]]))))
                        if candidate not in results:
                            results[candidate] = [nums[i], nums[j], nums[index]]

    keys = list(results.values())
    return keys


def get_next_str(s):
    tmp_s = ""
    dup = 1
    nums_list = []
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            dup += 1
        else:
            nums_list.append([s[i]] * dup)
            dup = 1
    if dup == 1:
        nums_list.append([s[i + 1]])
    else:
        nums_list.append([s[i + 1]] * dup)
    for nums in nums_list:
        tmp_s += str(len(nums)) + str(nums[0])
    return tmp_s


def search2(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """

    def find_peak(nums):
        i = 0
        j = len(nums) - 1
        mids_set = {}
        while i <= j:
            mid = ((j - i) // 2) + i
            if mid in mids_set:
                return -1
            mids_set[mid] = None
            if nums[mid] > nums[mid + 1] and nums[mid - 1] < nums[mid]:
                return mid
            if nums[mid] < nums[j]:
                j = mid
            if nums[mid] > nums[i]:
                i = mid
        return None

    def bin_search(nums, target):
        print(nums)
        left = 0
        right = len(nums) - 1
        mids_set = {}
        while left <= right:
            mid = left + (right - left) // 2
            if mid in mids_set:
                return -1
            mids_set[mid] = None
            if target == nums[mid]:
                return mid
            if target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    if not nums:
        return -1
    if len(nums) == 1:
        return 0 if nums[0] == target else -1
    peak_ind = find_peak(nums)
    if peak_ind == None:
        return -1
    left = bin_search(nums[:peak_ind + 1], target)
    right = bin_search(nums[peak_ind + 1:], target)
    print(peak_ind, left, right)
    if right == -1 and left == -1:
        return -1
    elif right == -1:
        return left
    else:
        return right + peak_ind + 1


def firstMissingPositive(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    candidate = 1
    num_d = {}
    max_num = -2 ** 31
    for num in nums:
        if num > max_num:
            max_num = num
        if num == candidate:
            candidate += 1
        num_d[num] = None
    if candidate not in num_d:
        return candidate
    while candidate in num_d:
        candidate += 1
    return candidate


def solution(A):
    # write your code in Python 3.6
    max_appeal = -2**31
    # for i in range(len(A)):
    #     for j in range(len(A)):
    #         appeal = A[i] + A[j] + i-j
    #         if  appeal > max_appeal:
    #             max_appeal = appeal
    # return max_appeal
    left = 0
    right = len(A) - 1
    while left < right:
        appeal = A[left] + A[right] + right - left
        if appeal > max_appeal:
            max_appeal = appeal
        appeal_left = A[left + 1] + A[right] + right - left + 1
        appeal_right = A[left] + A[right - 1] + right - 1 - left
        if appeal_left < appeal_right:
            right -= 1
        else:
            left += 1
    return max_appeal

if __name__ == '__main__':
    print(solution([1,3,-3] ))
    search2([3, 4, 5, 6, 1, 2], 2)
    # firstMissingPositive([2, 1])
    # get_next_str("1211")
    # subsets2([5, 4, 2, 7], 8)
    # threeSum2([0, 0, 0])
    # subsets([5, 4, 2, 7], 8)
    # backspaceCompare("a#b#", "ab##")
    # print(canCross([0,1,3,4,5,7,9,10,12]))
    # print(testing('yogev'))

    # test(1, 2, 3)
    # sec_biggest_elem([7, 2, 4, 6, 8, 90])
    # x = canCross([0, 1, 3, 5, 6, 8, 12, 17])
    # print(x)
    # maxAreaOfIsland([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])
    # longestPalindrome3("ac")
    # print(threeSum([-4, -2, 1, -5, -4, -4, 4, -2, 0, 4, 0, -2, 3, 1, -5, 0]))
    # maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7])
    # intToRoman(104)
    # isIntPalindrome(123)
    # print(isPalindrome("aa"))
    # print(meeting_order([[2, 11], [10, 30]]))
    #
    # moveZeroes([0, 0, 1])
    # maxProfit([7, 1, 5, 3, 6, 4])
    # print(numDecodings("123"))
    # subsets([1, 2, 3, 4])
    # print(match_strings_k_distinct("aBc", 'abcdb', 1))
    # removeDuplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])
    # divide(-2147483648, -1)
    # reverse_int(-123)
    # max_subs("aab")
    # print("hello")
    # letterCombinations("23")
# words = ["abc", "aab", "ayogev", "hello", "bla", 'boom', 'banana', 'aaaaaaa', "vwsdcvsdc"]
# ac = AutoComplete(words)
# songs = []
# for artist in range(50):
#     for song in range(random.randint(1, 5), random.randint(10, 15)):
#         songs.append((artist, song))
# shuffle_songs(songs)
# print(prime_factorization(48))
# prime_factorization_rec(48)
# s = "ab"
# p = ".*.."
# print(isMatch(s, p))
# print_fib(1,1,10000)
# create_arr_size_k_of_n(7, 5)
# n = 88
# fib_arr(n)
# print(fibo_loop(n))
# print(fibo(n))
# print_x(n)
# print(get_largest_delta([5, 99999, 23, 6, 8, 1, 7, 6, 4, 99, 9, 1, 0, 9, 8]))
# print(lotto([1, 3, 5, 2, 1, 5], 3, {}))
# print('*******')
# print(lotto2([_ for _ in range(1,44)], 6))
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
