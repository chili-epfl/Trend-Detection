"""
A modifiable GSP algorithm.
"""

class GspSearch(object):
    """
    A generic GSP algorithm, alllowing the individual parts to be overridden.

    This is setup so the object can be created once, but searched multiple times
    at different thresholds. In this generic form, we assume that the transactions
    are simply strings.
    """

    def __init__(self, raw_transactions):
        """
        C'tor, simply shaping the raw transactions into a useful form.
        """
        self.freq_patterns = []
        self.process_transactions(raw_transactions)

    def process_transactions(self, raw_transactions):
        """
        Create the alphabet & (normalized) transactions.
        """
        self.transactions = []
        alpha = {}
        for r in raw_transactions:
            for c in r:
                alpha[c] = True
            self.transactions.append(r)
        self.alpha = alpha.keys()

    def generate_init_candidates(self):
        """
        Make the initial set of candidate.
        Usually this would just be the alphabet.
        """
        return list(self.alpha)

    def generate_new_candidates(self, freq_pat):
        """
        Given existing patterns, generate a set of new patterns, one longer.
        """
        old_cnt = len(freq_pat)
        old_len = freq_pat[0].count(' ')
        print("Generating new candidates from %s %s-mers ..." % (old_cnt, old_len))

        new_candidates = []
        for c in freq_pat:
            for d in freq_pat:
                merged_candidate = self.merge_candidates(c, d)
                if merged_candidate and (merged_candidate not in new_candidates):
                    new_candidates.append(merged_candidate)

        ## Postconditions & return:
        return new_candidates

    def merge_candidates(self, a, b):
        c1 = a.split(' ')
        c2 = b.split(' ')
        if (len(c1) == 1 and len(c2) == 1) or c1[1:] == c2[:-1]:
            c1.append(c2[len(c2) - 1])
            return ' '.join(c1)
        else:
            return None

    def filter_candidates(self, trans_min):
        """
        Return a list of the candidates that occur in at least the given number of transactions.
        """
        filtered_candidates = []
        index = 0
        for c in self.candidates:
            curr_cand_hits = self.single_candidate_freq(c)
            if trans_min <= curr_cand_hits:
                filtered_candidates.append((c, curr_cand_hits))
            index+=1
        return filtered_candidates

    def single_candidate_freq(self, c):
        """
        Return true if a candidate is found in the transactions.
        """
        hits = 0
        for t in self.transactions:
            if self.search_transaction(t, c):
                hits += 1
        return hits

    def search_transaction(self, t, c):
        """
        Does this candidate appear in this transaction?
        """
        if c in t:
            return True
        t_len = len(t)
        found = False
        tokens = c.split(' ')
        found_total = len(tokens)
        for i in range(t_len):
            if tokens[0] == t[i]:
                found = True
                break
        if not found or i + found_total > t_len:
            return False
        for j in range(1, found_total):
            if tokens[j] != t[j + i]:
                return False
        return True

    def search(self, threshold):
        ## Preparation:
        assert (0.0 < threshold) and (threshold <= 1.0)
        trans_cnt = len(self.transactions)
        trans_min = trans_cnt * threshold

        print("The number of transactions is: %s" % trans_cnt)
        print("The minimal support is: %s" % threshold)
        print("The minimal transaction support is: %s" % trans_min)

        ## Main:
        # generate initial candidates & do initial filter
        self.candidates = list(self.generate_init_candidates())
        print("There are %s initial candidates." % len(self.candidates))
        freq_patterns = []
        new_freq_patterns = self.filter_candidates(trans_min)
        print("The initial candidates have been filtered down to %s." % len(new_freq_patterns))

        while True:
            # is there anything left?
            if new_freq_patterns:
                print(freq_patterns)
                freq_patterns = new_freq_patterns
                freq_terms = [x[0] for x in freq_patterns]
            else:
                self.freq_patterns = [x[0] for x in self.freq_patterns if ' ' in x[0]]
                print("The results are: ", self.freq_patterns)
                return self.freq_patterns

            # if any left, generate new candidates & filter
            self.candidates = self.generate_new_candidates(freq_terms)
            print("There are %s new candidates." % len(self.candidates))
            new_freq_patterns = self.filter_candidates(trans_min)
            new_freq_terms = [x[0] for x in new_freq_patterns]
            for term, freq in self.freq_patterns:
                found = False
                new_freq_term_index = 0
                while not found and new_freq_term_index < len(new_freq_terms):
                    if term in new_freq_terms[new_freq_term_index]:
                        found = True
                    else:
                        new_freq_term_index += 1
                if found:
                    self.freq_patterns.remove((term, freq))
            self.freq_patterns.extend(new_freq_patterns)
            print("The candidates have been filtered down to %s." % len(new_freq_patterns))