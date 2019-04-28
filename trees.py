import math
from random import randint


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


def contains(root, val):
    if val == root.val:
        return True
    if val > root.val:
        if not root.right:
            return False
        else:
            return contains(root.right, val)
    else:
        if not root.left:
            return False
        else:
            return contains(root.left, val)


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


def inorder_print(root):
    if root.left:
        inorder_print(root.left)
    print(root.val)
    if root.right:
        inorder_print(root.right)


def get_k_biggest(root, k):
    def append_preorder(root, arr, k):
        if len(arr) == k:
            return
        if root.right:
            append_preorder(root.right, arr, k)
        arr.append(root.val)
        if root.left:
            append_preorder(root.left, arr, k)

    arr = []
    append_preorder(root, arr, k)
    return arr[-1]


def get_k_smallest(root, k):
    def append_inorder(root, arr, k):
        if len(arr) == k:
            return
        if root.left:
            append_inorder(root.left, arr, k)
        arr.append(root.val)
        if root.right:
            append_inorder(root.right, arr, k)

    arr = []
    append_inorder(root, arr, k)
    return arr[-1]


# Lowest Common Ancestor Binary Tree

def LCA(root, a, b):
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
    if not root:
        return 0
    return max(find_depth(root.left), find_depth(root.right)) + 1


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


if __name__ == '__main__':
    root = Node(10)
    for _ in range(0, 20):
        insert(root, randint(0, 20))
    print(get_k_biggest(root, 3))
