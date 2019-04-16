


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
        print(f"Val {val} already exists")
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

def inorder_print(root):
    if root.left:
        inorder_print(root.left)
    print(root.val)
    if root.right:
        inorder_print(root.right)

root = Node(50)
for _ in range(0, 100):
    insert(root, randint(0, 100))

print(contains(root, 50))
inorder_print(root)
