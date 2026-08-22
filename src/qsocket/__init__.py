"""qsocket — hybrid classifier with an interchangeable 5->5 socket.

Model chain, identical across all arms:
    data -> PCA-5 -> scaling to [-pi/4, pi/4] -> [SOCKET 5->5] -> head -> score

Only the socket contents change (arms A-E). Everything before and after it is
bit-for-bit identical, because the estimand is the paired difference
delta = acc(A) - acc(B).
"""

__version__ = "0.1.0"
