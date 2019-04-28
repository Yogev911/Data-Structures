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


if __name__ == '__main__':

    head = ListNode(0)
    curr = head
    for _ in range(1, 2):
        curr.next = ListNode(_)
        curr = curr.next
    x = reverse_list(head)
    while x:
        print(x.val)
        x = x.next
