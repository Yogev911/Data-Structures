import random
import string
import math

BRACKETS = ['(', ')', '{', '}', '[', ']']


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

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


def is_valid_brackets(st):
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


def is_valid_brackets2(s):
    stack = []
    BRACKETS = ['(', ')', '{', '}', '[', ']']
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


def t(strs):
    i = 0

    if not strs:
        return ""
    while True:
        chars = []
        for word in strs:
            if not word:
                return ""
            if len(word) > i:
                chars.append(word[i])
            else:
                return word[:i]
        if len(set(chars)) != 1:
            return strs[0][:i]
        i += 1


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
    return num * sign


def find_start_char(char, st):
    '''
    complex = O(st)
    '''
    for i, c in enumerate(st):
        if c == char:
            return i
    return None


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


def get_closest_points(k, points, my_point):
    '''
    Complex = O(N + ) time
    Complex = O(N) space
    '''
    distances = {}
    for point in points:
        distance = get_distance(point, my_point)
        if distance not in distances:
            distances[distance] = [point]
        else:
            distances[distance].append(point)

    rel_distance = list(distances.keys())
    rel_distance.sort()
    ret_points = []
    for dis in rel_distance:
        for point in distances[dis]:
            if k == 0:
                return ret_points
            ret_points.append(point)
            k -= 1


def removeNthFromEnd(head, k):
    try:
        candidate = eval(f"head{'.next' * k}")
        if not candidate:
            candidate = eval(f"head{'.next' * (k - 2)}")
            candidate = None
            return head
    except:
        return False
    curr = head
    pre = curr
    curr = curr.next
    while curr.next:
        try:
            eval(f"pre{'.next' * k}")
        except:
            pre.next = curr.next
            return head
        curr = curr.next
        pre = pre.next


def removeNthFromEnd2(head, k):
    curr = head
    k_node = curr
    for _ in range(k):
        if curr.next:
            curr = curr.next
        else:
            return head.next
    return remove_node_element(k_node,curr)

def remove_node_element(k_node,curr):
    if not curr.next:
        k_node.next = k_node.next.next
        return head
    return remove_node_element(k_node.next,curr.next)

def mergeKLists(lists):
    if not any(lists):
        return None
    min_val, min_val_idx = get_val(lists)
    head = ListNode(min_val)
    curr_node = head
    lists[min_val_idx] = lists[min_val_idx].next
    while any(lists):
        min_val, min_val_idx = get_val(lists)
        curr_node.next = ListNode(min_val)
        curr_node = curr_node.next
        lists[min_val_idx] = lists[min_val_idx].next
    return head


def get_val(lists):
    min_val = None
    min_val_idx = None
    for i, list in enumerate([x for x in lists]):
        if list is not None:
            if min_val is None:
                min_val = list.val
                min_val_idx = i
            elif list.val < min_val:
                min_val = list.val
                min_val_idx = i
    return min_val, min_val_idx


if __name__ == '__main__':
    lists = []
    for i1 in range(0,random.randint(10,11)):
        head = ListNode(random.randint(0,100))
        curr = head
        for i2 in range(0,random.randint(3,5)):
            curr.next = ListNode(random.randint(0,100))
            curr = curr.next
        lists.append(head)

    print(lists)
    x = mergeKLists(lists)
    print('******************')
    while x:
        print(x.val)
        x = x.next
