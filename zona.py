def max_two_numbers_rec(lst):
    if len(lst) == 2:
        return lst[0], lst[1]

    else:
        m1, m2 = max_two_numbers_rec(lst[1:])
        if m1 > lst[0] and m2 > lst[0]:
            return m1, m2
        elif m1 > lst[0] and m2 < lst[0]:
            return m1, lst[0]
        elif m1 < lst[0] and m2 > lst[0]:
            return lst[0], m2


def exists_good_subset(lst, P, W, mem={}):
    if len(lst) == 0:
        if P <= 0 and W >= 0:
            return True
        else:
            return False

    if len(lst) in mem:
        return mem[len(lst)]
    res1 = exists_good_subset(lst[1:], P - lst[0][0], W - lst[0][1])
    res2 = exists_good_subset(lst[1:], P, W)
    mem[len(lst)] = res1 or res2
    return mem[len(lst)]
