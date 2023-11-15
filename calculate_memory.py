

if __name__ == '__main__':
    direction = 2
    patches = 49
    CHANNELS = 512
    N = 16
    N_SSM = 4
    mem_amount = direction*direction * patches * CHANNELS * N*N * N_SSM
    mem_amount_in_kb = int(mem_amount/1024)
    mem_amount_in_mb = int(mem_amount/(1024**2))
    mem_amount_in_gb = int(mem_amount/(1024**3))

    print(f'The memory amount is {mem_amount}b, {mem_amount_in_kb}kb, {mem_amount_in_mb}mb, {mem_amount_in_gb}gb')
