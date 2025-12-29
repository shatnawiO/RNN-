import numpy as np

def rnn_forward_pass(xt, h_prev, parameters):
 

    U = parameters["Wax"]   # (n_a, n_x)
    W = parameters["Waa"]   # (n_a, n_a)
    V = parameters["Wya"]   # (n_y, n_a)
    ba = parameters["ba"]   # (n_a, 1)
    by = parameters["by"]   # (n_y, 1)

    # Compute next hidden state a_t (you called it ht)
    ht = np.tanh(np.dot(U, xt) + np.dot(W, h_prev) + ba)   # (n_a, 1)

    # Compute output y_t
    yt = np.dot(V, ht) + by   # (n_y, 1)

    # ---- IMPORTANT: cache ALL values needed for backward ----
    cache = (ht, h_prev, xt, parameters)

    return ht, yt, cache




def rnn_backward_pass(dht, cache):
   

    ht, h_prev, xt, parameters = cache

    U = parameters["Wax"]   # (n_a, n_x)
    W = parameters["Waa"]   # (n_a, n_a)
    V = parameters["Wya"]   # (n_y, n_a)

    # dtanh = (1 - a_t^2)
    dtanh = (1 - ht ** 2) * dht   # (n_a, 1)

    # Gradients
    dWax = np.dot(dtanh, xt.T)       # (n_a, n_x)
    dWaa = np.dot(dtanh, h_prev.T)   # (n_a, n_a)
    dba = dtanh                      # (n_a, 1)

    dx = np.dot(U.T, dtanh)          # (n_x, 1)
    dh_prev = np.dot(W.T, dtanh)     # (n_a, 1)

    grads = {
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba,
        "dx": dx,
        "dh_prev": dh_prev
    }

    return grads



def rnn_forward(x, h0, parameters):

    caches = []

    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    h = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    h_next = h0

    for t in range(T_x):

        xt = x[:, :, t]     # (n_x, m)

        # Loop over batch samples
        ht_all = []
        yt_all = []
        cache_all = []

        for i in range(m):
            ht, yt, cache = rnn_forward_pass(xt[:, i:i+1], h_next[:, i:i+1], parameters)
            ht_all.append(ht)
            yt_all.append(yt)
            cache_all.append(cache)

        h[:, :, t] = np.hstack(ht_all)
        y_pred[:, :, t] = np.hstack(yt_all)

        caches.append(cache_all)

        h_next = h[:, :, t]

    return h, y_pred, (caches, x)


def rnn_backwards(dh, caches):

    (caches, x) = caches

    # Get shapes
    (ht, h_prev, xt, parameters) = caches[0][0]
    n_a, n_x = parameters["Wax"].shape
    n_x, m, T_x = x.shape

    # Initialize gradients
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba  = np.zeros((n_a, 1))
    dh_prev_t = np.zeros((n_a, m))

    # Loop backward
    for t in reversed(range(T_x)):

        for i in range(m):
            dht = dh[:, i:i+1, t] + dh_prev_t[:, i:i+1]

            grads = rnn_backward_pass(dht, caches[t][i])

            dWax += grads["dWax"]
            dWaa += grads["dWaa"]
            dba  += grads["dba"]

            dx[:, i:i+1, t] = grads["dx"]
            dh_prev_t[:, i:i+1] = grads["dh_prev"]

    return {
        "dx": dx,
        "dWax": dWax,
        "dWaa": dWaa,
        "dba": dba,
        "dh0": dh_prev_t
    }




    

    


