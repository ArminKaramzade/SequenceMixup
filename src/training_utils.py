import numpy as np

def pad_sequence(sequences, batch_first=True, padding_value=0, padding='post'):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    if padding == 'post':
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, :length, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
    elif padding == 'pre':
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            if batch_first:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[-length:, i, ...] = tensor
    else:
        raise ValueError("Padding must be 'post' or 'pre'")
    return out_tensor

def sort(c):
    a, b = c
    idx = [i[0] for i in sorted(enumerate(a), key=lambda s: len(s[1]), reverse=True)]
    return ([a[i] for i in idx], [b[i] for i in idx])
      
    
def beta_lambdas_generator(alpha, beta, batch_size, length, repeat, rho):    
    def extend(a, repeat):
        # a.shape = (batch_size, length)
        # repeat = (n1, n2, ...)
        a = np.tile(a, repeat+(1, 1)) # ((repeat), batch_size, length)
        a = np.rollaxis(a, len(a.shape)-2, 0) # (batch_size, (repeat), length)
        return np.rollaxis(a, len(a.shape)-1, 1) # (batch_size, length, (repeat))
    
    def get_ab(alpha, beta, rho, x):
        c1 = rho * (alpha / (alpha + beta)) + (1 - rho) * (x)
        c2 = (rho**2) * (alpha*beta) / (((alpha + beta)**2) * (alpha + beta + 1))
        if c2 == 0:
            a = 1e9
        else:
            a = (c1 * (1 - c1) - c2) * c1 / c2
        if c1 == 0:
            b = 1e9
        else:
            b = a * (1. / c1 - 1)
        return max(1e-9, a), max(1e-9, b)
    
    if rho == 0:
        lambdas = np.random.beta(alpha, beta, (batch_size))
        lambdas = np.tile(lambdas, (length, 1))
        lambdas = np.rollaxis(lambdas, len(lambdas.shape)-1, 0)
        return extend(lambdas, repeat)

    lambdas = np.zeros((batch_size, length))

    for i in range(batch_size):
        for j in range(length):
            if j == 0:
                lambdas[i, j] = np.random.beta(alpha, beta)
            else:
                a, b = get_ab(alpha, beta, rho, lambdas[i, j-1])
                lambdas[i, j] = np.random.beta(a, b)
                
    return extend(lambdas, repeat)