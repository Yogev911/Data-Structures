def rotate_by_2(lst, rotates=None):
    if not lst:
        return []
    rotates = rotates % len(lst)
    if rotates == 0:
        return lst
    if len(lst) <= rotates:
        return lst

    return lst[-rotates:] + lst[:-rotates]


def fixed_point(i, arr, hmap):
    '''
    implement a method that given a sorted array with negative and positive values,
    find a value that is equal to its index.
    for example: [1, 2, 4, 5, 4] => will return 4. Do it with better complexity than O(n).
    '''
    if not arr or i in hmap:
        return "Not Found"
    hmap[i] = None
    # if len(arr) == 1:
    #     print(arr[0])
    if i == arr[i]:
        return i
    if i > arr[i]:
        return fixed_point(int(i + len(arr[i:]) / 2), arr, hmap)
    if i < arr[i]:
        return fixed_point(int(i - len(arr[:i]) / 2), arr, hmap)


if __name__ == '__main__':
    print(rotate_by_2([1, 2, 3, 4, 5, 6, 7, 8, 9], 99999999))
