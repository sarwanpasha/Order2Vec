
import sys
from collections import defaultdict

def compute_minimizers(s, k, m, debug = False) :
    minimizers = defaultdict(int) # frequency vector

    queue = []
    mi = None # index in queue of the current minimizer
    for i in range(len(s) - k + 1) :
        kmer = s[i:i+k]

        if mi :

            queue.pop(0) # out with the old
            mmer = s[i+k-m:i+k] # in with the new
            mi -= 1 # shift index back

            mmer = min(mmer, mmer[::-1]) # lexicographically smallest forward/reverse
            queue.append(mmer)

            if mmer < queue[mi] : # update with new
                mi = k-m

        else :
            queue = [] # reset the queue, start from scratch

            mi = 0 # first m-mer
            for j in range(k - m + 1) :
                mmer = kmer[j:j+m]
                mmer = min(mmer, mmer[::-1])
                queue.append(mmer)

                if mmer < queue[mi] : # keep track of current minimizer
                    mi = j

        minimizers[queue[mi]] += 1 # update frequency vector

        if debug :
            print(kmer, '->', queue[mi])

    return minimizers

# Main
#----------------------------------------------------------------------

s = sys.argv[1]
k = int(sys.argv[2])
m = int(sys.argv[3])

debug = True if len(sys.argv) > 4 else False
if debug :
    print('python3', sys.argv[0], s, k, m, sys.argv[4])

minimizers = compute_minimizers(s, k, m, debug)

if debug :
    print(minimizers)
