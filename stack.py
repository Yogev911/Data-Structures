import random
import string
import math

from heap import MaxHeap


class SmartStack:
    def __init__(self):
        self._stack = []
        self._min = []
        self._max = []

    def __str__(self):
        return ', '.join([str(x) for x in self._stack])

    def push(self, x):
        self._stack.append(x)
        if len(self._min) != 0:
            if x < self.get_min():
                self._min.append(x)
        else:
            self._min.append(x)
        if len(self._max) != 0:
            if x > self.get_max():
                self._max.append(x)
        else:
            self._max.append(x)

    def pop(self):
        x = self._stack.pop()
        if x == self.get_min():
            self._min.pop()
        if x == self.get_max():
            self._max.pop()
        return x

    def get_min(self):
        return self._min[-1]

    def get_max(self):
        return self._max[-1]

    def pop_max(self):
        return self._max.pop()


class Stack:
    def __init__(self):
        self._stack = []
        self._size = 0

    def __str__(self):
        return ', '.join([str(x) for x in self._stack])

    def isEmpty(self):
        return not bool(self._size)

    def push(self, item):
        self._stack.append(item)
        self._size += 1

    def pop(self):
        if self._size:
            self._size -= 1
            return self._stack.pop()
        raise BufferError("Empty stack")

    def peek(self):
        if self._size:
            return self._stack[self._size - 1]
        raise BufferError("Empty stack")

    def get_size(self):
        return self._size


def balanced_parentheses(st):
    BRACKETS = ['(', ')', '{', '}', '[', ']']
    stack = Stack()
    try:
        for char in st:
            if char in BRACKETS:
                if char == ')':
                    if stack.peek() == '(':
                        stack.pop()
                elif char == ']':
                    if stack.peek() == '[':
                        stack.pop()
                elif char == '}':
                    if stack.peek() == '{':
                        stack.pop()
                else:
                    stack.push(char)
        if not stack.isEmpty():
            return False
        return True
    except BufferError:
        return False


def balanced_parentheses2(s):
    # no need stack object
    BRACKETS = ['(', ')', '{', '}', '[', ']']
    stack = []
    for char in s:
        if char in BRACKETS:
            if char == ')':
                if not stack:
                    return False
                if stack[-1] == '(':
                    stack.pop()
                else:
                    stack.append(char)
            elif char == ']':
                if not stack:
                    return False
                if stack[-1] == '[':
                    stack.pop()
                else:
                    stack.append(char)
            elif char == '}':
                if not stack:
                    return False
                if stack[-1] == '{':
                    stack.pop()
                else:
                    stack.append(char)
            else:
                stack.append(char)
    if stack:
        return False
    return True


def get_dupes(a):
    '''
    split list to duplicates and seen
    :param a:
    :return:
    '''
    seen = {}
    dupes = []
    for x in a:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes, seen


def lengthOfLongestSubstring(s):
    '''
    Given a string, find the length of the longest substring without repeating characters.
    :param s:
    :return:
    '''
    max_subs = 0
    for i in range(0, len(s)):
        d = {}
        for char in s[i:]:
            if char not in d:
                d[char] = None
            else:
                break
        curr_size = len(d.keys())
        if curr_size > max_subs:
            max_subs = curr_size
    return max_subs


def myAtoi(str):
    str = str.lstrip()
    if not str:
        return 0
    stack = []
    sign = 1
    if not str[0].isdigit():
        if str[0] == '-':
            sign = -1
            str = str[1:]
        elif str[0] == '+':
            sign = 1
            str = str[1:]
        else:
            return 0
    while str:
        if str[0].isdigit():
            stack.append(str[0])
            str = str[1:]
        else:
            break
    num = 0
    i = 0
    while stack:
        right_num = stack.pop()
        num += int(right_num) * 10 ** i
        i += 1
    num = num * sign
    if -2 ** 31 > num:
        return -2 ** 31
    if num > 2 ** 31 - 1:
        return 2 ** 31 - 1
    return num


def get_common_sub(s1, s2):
    '''
    Complex = O(s1*s2)
    '''
    if not s1 and s2:
        return ""
    stack = []
    d = {}
    start = 0
    for i1, char1 in enumerate(s1):
        if len(s2) <= start:
            "".join(stack)
        next_char_idx = find_start_char(char1, s2[start:])
        if next_char_idx is not None:
            stack.append(char1)
            start += next_char_idx + 1
            continue
        else:
            continue
    return "".join(stack)


def find_start_char(char, st):
    '''
    complex = O(N)
    '''
    for i, c in enumerate(st):
        if c == char:
            return i
    return None


def get_longest_common_sub(s1, s2):
    '''
    Complex = O(s1^2*s2^2)
    :param s1:
    :param s2:
    :return:
    '''
    subs = {}
    for range1 in range(0, len(s1)):
        for range2 in range(0, len(s2)):
            sub = get_common_sub(s1[range1:], s2[range2:])
            if len(sub) not in subs:
                subs[len(sub)] = sub
    return subs[max(subs.keys())]


def get_distance(p1, p2):
    '''
    Complex = O(1) time
    Complex = O(1) space

    '''
    return math.sqrt(abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2)


def get_closest_points(k,my_point, points):
    '''
    Complex = O(N + ) time
    Complex = O(N) space
    '''
    distances = {}
    s = MaxHeap([])
    for point in points:
        distance = get_distance(point, my_point)
        if distance not in distances:
            distances[distance] = [point]
            s.push(distance)
        else:
            distances[distance].append(point)
    ret_points = []
    while k:
        ret_points.append(distances[s.pop()])
        k -= 1
    return ret_points
    # rel_distance = list(distances.keys())
    # rel_distance.sort()
    #
    # for dis in rel_distance:
    #     for point in distances[dis]:
    #         if k == 0:
    #             return ret_points
    #         ret_points.append(point)
    #         k -= 1


def perform_postfix(st):
    '''
    time complex O(N)
    space complex O(N)
    perform matematic infix
    :param st: str ex. '56+6*73+3/-1+'
    :return: int ex . 64
    '''
    signs = {'+': lambda y, x: x + y,
             '-': lambda y, x: x - y,
             '/': lambda y, x: x / y,
             '*': lambda y, x: x * y}
    s = Stack()
    for char in st:
        if char.isdigit():
            s.push(char)
        else:
            s.push(signs[char](int(s.pop()), int(s.pop())))
    return s.pop()


def infix_to_postfix(st):
    '''
    time complex O(N)
    space complex O(N)
    perform matematic infix
    :param st: str ex. '(5+6)*6-(7+3)/3+1'
    :return: str ex . '56+6*73+3/-1+'
    '''
    ranks = {'+': 1,
             '-': 1,
             '/': 2,
             '*': 2,
             '(': 0,
             ')': 0}
    s = Stack()
    postfix = ''
    for char in st:
        if char.isdigit():
            postfix += char
        else:
            if s.isEmpty():
                s.push(char)
            else:
                if char == '(':
                    s.push(char)
                elif char == ')':
                    while s.peek() != '(':
                        postfix += s.pop()
                    s.pop()
                elif ranks[s.peek()] < ranks[char]:
                    s.push(char)
                else:
                    while not s.isEmpty():
                        postfix += s.pop()
                    s.push(char)
    while not s.isEmpty():
        postfix += s.pop()
    print(perform_postfix(postfix))


if __name__ == '__main__':
    get_closest_points(5,(4,7),[(random.randint(1,10),random.randint(1,10)) for i in range(10)])
