import random


def decToBin(x):
    """
    十进制小数转二进制小数
    :param x: 十进制小数
    :return: str
    """
    bins = []
    if x < 0:
        x = -x
        bins.append('-')
    integer = int(x)
    if integer > 0:
        bins.append(bin(integer)[2:])
    else:
        bins.append('0')
    bins.append('.')
    decimal = x - integer
    while decimal:
        decimal *= 2
        bins.append('1' if decimal >= 1. else '0')
        decimal -= int(decimal)
    # print(bins)
    res = ''.join(bins)
    return res


def binToDec(b):
    """
    二进制小数转十进制小数
    :param b: 二进制小数字符串
    :return: float
    """
    isPositive = True # 是否是正数
    if b[0] == '-':
        isPositive = False
        b = b[1:]
    integer, decimal = b.split('.')

    int_dec = 0 # 整数部分的十进制
    for i, x in enumerate(reversed(integer)):
        int_dec += int(x) * 2 ** (i)
    dec_dec = 0 # 小数部分的十进制
    for i, x in enumerate(decimal):
        dec_dec += int(x) * 2 ** (-i-1)
    # print(int_dec, dec_dec)
    res = int_dec + dec_dec
    return res if isPositive else -res


def changeOneBit(b):
    """
    修改二进制小数其中一位
    :param b:
    :return:
    """
    for i in range(len(b)):
        if b[i] == '1':
            change_idx = random.randint(i, len(b)-1)
            if b[change_idx] == '.':
                change_idx += 1
            change = '1' if b[change_idx] == '0' else '0'
            if change_idx != len(b)-1:
                res = b[:change_idx] + change + b[change_idx+1:]
            else:
                res = b[:change_idx] + change
            break
    return res


def changeOneBitInBinary(x):
    return binToDec(changeOneBit(decToBin(x)))



if __name__ == '__main__':

    # print(dec2bin(0.8125))
    # # [1, 1, 0, 1]
    # print(bin2dec(dec2bin(0.8125)))
    # # 0.8125

    print(decToBin(-0.0008112))
    print(binToDec(decToBin(-0.0008112)))
    print(changeOneBitInBinary(-0.0008112))
