def rotate_by_2(lst, rotates=None):
    if not lst:
        return []
    rotates = rotates % len(lst)
    if rotates == 0:
        return lst
    if len(lst) <= rotates:
        return lst

    return lst[-rotates:] + lst[:-rotates]


if __name__ == '__main__':
    print(rotate_by_2([1, 2, 3, 4, 5, 6, 7, 8, 9], 99999999))
