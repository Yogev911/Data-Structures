import random


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def reverse_list(head):
    curr = head.next
    head.next = None

    while curr:
        tmp = curr
        curr = curr.next
        tmp.next = head
        head = tmp
    return head


def reverse_list_rec(head, k):
    if not (head and head.next) or k == 0:
        return head
    rev_head = reverse_list_rec(head.next, k - 1)
    head.next.next = head
    head.next = None
    return rev_head


def reverseKGroup(head, k):
    """
    :type head: ListNode
    :type k: int
    :rtype: ListNode
    """
    curr = head
    dummy = ListNode(None)
    ret_head = dummy
    next_g = curr
    has_g = True
    while has_g:
        for _ in range(k):
            if next_g:
                next_g = next_g.next
            else:
                next_g = curr
                has_g = False
                break
        if has_g:

            dummy.next = reverse_list_rec(curr, k-1)
            while dummy != curr:
                dummy = dummy.next
            # dummy = dummy.next
            curr = next_g
        else:
            dummy.next = curr
            break

    return ret_head.next


def removeNthFromEnd(head, k):
    curr = head
    k_node = curr
    for _ in range(k):
        if curr.next:
            curr = curr.next
        else:
            return head.next
    return remove_node_element(k_node, curr)


def remove_node_element(k_node, curr):
    if not curr.next:
        k_node.next = k_node.next.next
        return head
    return remove_node_element(k_node.next, curr.next)


def get_k_element_from_ll(head, k):
    k_node = head
    for _ in range(k):
        if head:
            head = head.next
        else:
            return False
    while head:
        head = head.next
        k_node = k_node.next
    return k_node.val


def get_k_element_from_ll_rec(head, k):
    k_node = head
    for _ in range(k):
        if head:
            head = head.next
        else:
            return False
    return k_elem_helper(head, k_node)


def k_elem_helper(head, k_node):
    if not head:
        return k_node.val
    return k_elem_helper(head.next, k_node.next)


def mergeKLists(lists):
    if not any(lists):
        return None
    min_val, min_val_idx = get_node_param(lists)
    head = ListNode(min_val)
    curr_node = head
    lists[min_val_idx] = lists[min_val_idx].next
    while any(lists):
        min_val, min_val_idx = get_node_param(lists)
        curr_node.next = lists[min_val_idx]
        curr_node = curr_node.next
        lists[min_val_idx] = lists[min_val_idx].next
    return head


def get_node_param(lists):
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


def swapPairs(head):
    head_new = ListNode(None)
    head_new.next = head
    curr = head_new
    while curr.next and curr.next.next:
        tmp = curr.next
        curr.next = curr.next.next
        tmp.next = tmp.next.next
        curr.next.next = tmp
        curr = curr.next.next
    return head_new.next
    # def swap(curr):
    #     tmp = curr.next.next
    #     new_head = curr.next
    #     new_head.next = curr
    #     new_head.next.next = tmp
    #     return new_head
    #
    # curr = head
    # cuur = ListNode(777)
    # t = cuur
    # curr.next = head
    # curr = curr.next
    # # new_h = swap(curr)
    # while curr.next.next:
    #     tmp = curr.next.next
    #     new_head = curr.next
    #     new_head.next = curr
    #     new_head.next.next = tmp
    #     new_head = new_head.next.next
    #     curr = new_head
    #     # curr = curr.next.next
    # return t.next


def addTwoNumbers(l1, l2):
    dummy = ListNode(None)
    curr = dummy
    acc = 0
    carry = 0
    while l1 or l2:
        if not l1:
            carry += l2.val
            l2 = l2.next
        elif not l2:
            carry += l1.val
            l1 = l1.next
        else:
            carry += l1.val + l2.val
            l2 = l2.next
            l1 = l1.next
        if carry > 9:
            acc = carry % 10
            carry //= 10
        else:
            acc = carry
        curr.next = ListNode(acc)
        curr = curr.next
        acc = 0
    if carry:
        curr.next = ListNode(carry)
    return dummy.next


def create_ll(RANDOM=False):
    global head, x, curr
    head = ListNode(1)
    x = head
    curr = head
    for _ in range(2, 7):
        if RANDOM:
            curr.next = ListNode(random.randint(1, 10))
        else:
            curr.next = ListNode(_)
        curr = curr.next
    return head


def addTwoNumbers(l1, l2):
    dummy = ListNode(None)
    curr = dummy
    carry = 0
    while l1 or l2:
        if not l1:
            carry += l2.val
            l2 = l2.next
        elif not l2:
            carry += l1.val
            l1 = l1.next
        else:
            carry += l1.val + l2.val
            l2 = l2.next
            l1 = l1.next
        if carry > 9:
            acc = carry % 10
            carry //= 10
        else:
            acc = carry
            carry = 0
        curr.next = ListNode(acc)
        curr = curr.next
    if carry:
        curr.next = ListNode(carry)
    return dummy.next


'''
public class SolutionIterator {

public SolutionIterator(ImmutableList < ImmutableList < Integer >> lists) {

}

public int next() {

}

public boolean hasNext() {

}
}
'''

import heap


class SolutionIterator:
    def __init__(self, N):
        self.my_gen = self._generate_sort(N)
        self.size = sum([len(lst) for lst in N])

    def _generate_sort(self, N):
        list_match = None
        while any(N):
            min_val = 2 ** 31
            for lst in N:
                if lst and lst[0] <= min_val:
                    min_val = lst[0]
                    list_match = lst
            list_match.pop(0)
            yield min_val

    def get_next(self):
        self.size -= 1
        return next(self.my_gen) if self.size + 1 else "End of iteration"

    def has_next(self):
        return True if self.size else False

    def isEmpty(self):
        return not bool(self.size)


def uniquePaths(m, n):
    paths = [[0 for x in range(n)] for y in range(m)]
    for i in range(len(paths)):
        paths[i][0] = 1
    for i in range(len(paths[0])):
        paths[0][i] = 1
    for i in range(1, len(paths)):
        for j in range(1, len(paths[i])):
            paths[i][j] = paths[i - 1][j] + paths[i][j - 1]
    return paths[m - 1][n - 1]


def isHappy(n, seen={}):
    if n == 1:
        return True
    if n in seen:
        return False
    seen[n] = None
    num = sum([int(d) ** 2 for d in str(n)])
    return isHappy(num, seen)


def merge(intervals):
    if not intervals:
        return []
    min_val = min([i[0] for i in intervals])
    max_val = max([i[1] for i in intervals])
    arr = [None] * (max_val - min_val + 1)
    for i in range(len(intervals)):
        for index in range(intervals[i][0], intervals[i][1] + 1):
            arr[index - min_val] = 1

    for i in range(len(arr)):
        if arr[i]:
            arr[i] = i + min_val
    start = 0
    end = 0
    output = []
    for i in range(len(arr)):
        if not arr[i]:
            if not arr[i - 1]:
                start = i + 1
                continue
            output.append([arr[start], arr[i - 1]])
            start = i + 1
    output.append([arr[start], arr[i]])
    print(arr)
    return output


def canConstruct(ransomNote, magazine):
    char_map = {}
    for char in magazine:
        if char not in char_map:
            char_map[char] = 1
            char_map[char] += 1
    for char in ransomNote:
        if char not in char_map:
            return False
        elif char_map[char] == 0:
            return False
        else:
            char_map[char] -= 1
    return True


if __name__ == '__main__':
    # canConstruct("aa", "ab")
    # merge([[1, 4], [5, 6]])
    # print(isHappy(10))
    # uniquePaths(3, 7)
    # N = [
    #     [2, 3, 6, 9],
    #     [1, 4, 7],
    #     [5, 8]
    # ]
    # x = SolutionIterator(N)
    # while not x.isEmpty():
    #     print(x.has_next())
    #     print(x.get_next())
    l1 = create_ll()
    x = reverseKGroup(l1, 3)
    print(x)
    # l2 = create_ll()
    # x = addTwoNumbers(l1, l2)
