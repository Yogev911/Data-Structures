import math
import random
from random import randint

from tests import timer


class Node(object):
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val

    # def __str__(self):
    #     print_inorder(self)


def insert(root, val):
    if val == root.val:
        # print(f"Val {val} already exists")
        return
    if val > root.val:
        if not root.right:
            root.right = Node(val)
        else:
            insert(root.right, val)
    else:
        if not root.left:
            root.left = Node(val)
        else:
            insert(root.left, val)


def get_rank(root, val):
    if not root:
        raise ValueError(f"val {val} is not found in tree")
    if root.val == val:
        return 0
    if val > root.val:
        return 1 + get_rank(root.right, val)
    if val < root.val:
        return 1 + get_rank(root.left, val)


def get_parent(child):
    if not child.left and not child.right:
        return None
    if child.left and child.val == child.left.val:
        return child
    if child.right and child.val == child.right.val:
        return child

    left = get_parent(child)
    right = get_parent(child)
    return left if left else right


def preorder_print(root):
    if root:
        print(root.val)
    else:
        return
    if root.left:
        preorder_print(root.left)
    if root.right:
        preorder_print(root.right)


def inorder_print(root):
    if root.left:
        inorder_print(root.left)
    print(root.val)
    if root.right:
        inorder_print(root.right)


def postorder_print(root):
    if not root:
        return
    if root.left:
        postorder_print(root.left)
    if root.right:
        postorder_print(root.right)
    print(root.val)


def get_k_biggest(root, arr, k):
    if len(arr) >= k:
        print(arr[k - 1])
        return
    if root.right:
        get_k_biggest(root.right, arr, k)
    arr.append(root.val)
    if root.left:
        get_k_biggest(root.left, arr, k)


def get_k_smallest(root, arr, k):
    if len(arr) >= k:
        print(arr[k - 1])
        return
    if root.left:
        get_k_smallest(root.left, arr, k)
    arr.append(root.val)
    if root.right:
        get_k_smallest(root.right, arr, k)


def find_k_element2(root, k, arr):
    if root.right:
        find_k_element2(root.right, k, arr)
    arr.append(root.val)
    if len(arr) == k:
        return
    if root.left:
        find_k_element2(root.left, k, arr)


def LCA(root, a, b):
    # Lowest Common Ancestor Binary Tree
    if root is None:
        return None
    if root.val == a or root.val == b:
        return root
    left = LCA(root.left, a, b)
    right = LCA(root.right, a, b)
    if left and right:
        return root
    return left if left else right


def find_depth(root):
    return max(find_depth(root.left), find_depth(root.right)) + 1 if root else 0


def is_balanced(root):
    if not root:
        return 0
    left = is_balanced(root.left)
    if left == -1:
        return -1
    right = is_balanced(root.right)
    if right == -1:
        return -1
    if abs(left - right) > 1:
        return -1
    return 1 + max(right, left)


def contains(root, val):
    if not root:
        return False
    if root.val == val:
        return True
    if val > root.val:
        return contains(root.right, val)
    if val < root.val:
        return contains(root.left, val)


def tree2str(t):
    if not t:
        return ""
    if not t.left and not t.right:
        return str(t.val)
    left = tree2str(t.left)
    right = tree2str(t.right)
    if right == '':
        return f"{t.val}({left})"
    else:
        return f"{t.val}({left})({right})"


def str2tree(s):
    root = Node(s[0])
    s = s[1:]

    if s == '':
        return


if __name__ == '__main__':
    pass
    # arr = [-500, -300, -200, -100, 0, 1, 2, 3, 4, 5, 6, 10, 11, 13, 45, 78, 9999]
    # print(fixed_point(int(len(arr) / 2), arr, {}))
    #
    root = Node(5)
    for _ in range(1, 10):
        insert(root, random.randint(1,10))
    inorder_print(root)
    print('^^^^^^^^^')
    preorder_print(root)
    print('^^^^^^^^^')
    postorder_print(root)
    print('')
    # print(contains2(root, 999))
    # print(get_rank(root, 9999))
    # print(arr)
    # print(contains2(root,11))
    # insert(root, 15)
    # insert(root, 13)
    # insert(root, 2)
    # insert(root, 5)

    # print(is_balanced(root))
    # print(get_k_biggest(root, 3))
