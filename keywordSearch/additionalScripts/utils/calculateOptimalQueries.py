import pickle, itertools
from tqdm import tqdm

MAX_QUERIES = 100  # Amount of queries to save at optimal. If x is the optimal RR, then only the first MAX_QUERIES queries


# that achieve X RR will be saved (cuts down on the amount of queries saved)

def calculateOptimalQueries(dataset, sender, receiver, oracle, start_range, end_range, min_length, max_length,
                            save_path, top_k=20,
                            sample_set=None):
    def getRR(matchingIDs, returnedIDs):
        rr = 0
        for match in matchingIDs:
            if match in returnedIDs and (1 / (returnedIDs.index(match) + 1)) > rr:
                rr = 1 / (returnedIDs.index(match) + 1)

        return rr

    best_queries = {}

    if sample_set is None:
        sample_set = sender.strategy.signalIndex.keys()

    for i, tupID in enumerate(tqdm([x for x in sample_set][start_range:end_range])):

        best_queries[tupID] = [[] for i in range((max_length - min_length) + 1)]
        best_rr = [0] * ((max_length - min_length) + 1)

        if tupID in oracle.data1:
            matchingIDs = [id for id in oracle.data1[tupID] if id in receiver.signalIndex]
        elif tupID in oracle.data2:
            matchingIDs = [id for id in oracle.data2[tupID] if id in receiver.signalIndex]
        else:
            print(f'No match for {tupID}--continue')
            continue

        # Get keywords from matches on external
        reciever_match_keywords = set(
            [k.keyword for matchID in matchingIDs for k in receiver.signalIndex[matchID]])

        # Get keywords shared by local and external tuples
        overlapping_keywords = []
        nonoverlapping_keywords = []
        for signal in sender.signalIndex[tupID]:
            processed_keyword = signal.keyword
            if processed_keyword.lower() in reciever_match_keywords:
                overlapping_keywords.append(signal)
            else:
                nonoverlapping_keywords.append(signal)

        overlapping_keywords = sorted(overlapping_keywords, key=(lambda x: x.keyword))
        nonoverlapping_keywords = sorted(nonoverlapping_keywords, key=(lambda x: x.keyword))

        for query_size in range(min_length, max_length + 1):

            index = query_size - min_length

            full_overlap_queries = [query for query in itertools.combinations(overlapping_keywords, r=query_size)]

            # Compute all queries using only shared keywords
            for keywords_to_send in full_overlap_queries:

                # Get results using this query
                returnedIDs = receiver.returnTuples([(signal, signal.keyword) for signal in keywords_to_send], top_k)
                rr = getRR(matchingIDs, returnedIDs)

                # This one is better than anything we have found thus far. Dump the current list and start a new one.
                if best_rr[index] < rr:
                    best_rr[index] = rr
                    best_queries[tupID][index] = []
                    best_queries[tupID][index].append((keywords_to_send, rr))
                # This one performs the same as the best ones we have found upto this point. Append it.
                elif 0 < best_rr[index] <= rr and len(best_queries[tupID][index]) < MAX_QUERIES:
                    best_queries[tupID][index].append((keywords_to_send, rr))
                    if len(best_queries[tupID][index]) == MAX_QUERIES and best_rr[index] == 1:
                        break

            # Compute best RR using all keywords. Note that using non-overlapping keywords can never increase our RR
            #   over queries which only use overlapping keywords... This is a check to see if we can add "throwaway"
            #   keywords that are only added to meet the query length requirement.
            #    Skip cases where
            #     - index == 0: queries of size 1 that include non-overlapping keywords will do nothing
            #     - best_rr[index-1] <= best_rr[index]: we either saw an improvement from adding an extra
            #     overlapping keyword, or we saw no change. Either way, we know a query containing non-overlapping terms
            #     <= best_rr[index-1]. So we can't do better
            if index != 0 and best_rr[index - 1] > best_rr[index]:
                best_queries_filler_queries = []
                for bestPreviousQuery in best_queries[tupID][index - 1]:

                    for keyword_to_add in nonoverlapping_keywords:
                        if keyword_to_add not in bestPreviousQuery[0]:
                            keywords_to_send = bestPreviousQuery[0] + (keyword_to_add,)
                        else:
                            continue

                        # Get results using this query
                        returnedIDs = receiver.returnTuples([(signal, signal.keyword) for signal in keywords_to_send],
                                                            top_k)
                        rr = getRR(matchingIDs, returnedIDs)

                        # Keep track of first best query using all keywords
                        if best_rr[index] < rr:
                            best_rr[index] = rr
                            best_queries_filler_queries = []
                            best_queries_filler_queries.append((keywords_to_send, rr))

                        # Can't do better than previous best by adding a keyword that doesn't exist in the match, so stop early
                        if best_rr[index - 1] == best_rr[index]:
                            break

                best_queries[tupID][index] += best_queries_filler_queries

            # This is for the case where our first length (min_length) isn't 1. In this case, its possible that we can
            # do better than the RR gotten from sending min_length overlapping keywords by sending (min_length - n)
            # overlapping keywords and n non-overlapping keywords
            elif index == 0 and min_length > 1 and best_rr[index] < 1:

                # Remove all but (query_size - 1) terms that do not overlap with the external INDEX
                # For any two terms a, b; if a,b not in D, then a == b for all practical purposes
                    # Find all keywords not in D; remove then from set and add to different set
                nonoverlapping_keywords_D = [signal for signal in nonoverlapping_keywords if signal.keyword not in receiver.idf]
                    # remove all of these from nonoverlapping_keywords
                nonoverlapping_keywords = list(set(nonoverlapping_keywords).difference(nonoverlapping_keywords_D))
                    # add (query_size - 1) back in
                nonoverlapping_keywords += nonoverlapping_keywords_D[:query_size - 1]

                all_keywords = overlapping_keywords + nonoverlapping_keywords
                best_queries_filler_queries = []

                # Only try queries that include at least one keyword that overlaps with an external match
                partial_overlap_queries = [query for query in itertools.combinations(all_keywords, r=query_size) if len(set(query).intersection(overlapping_keywords)) > 0 and query not in full_overlap_queries]

                for keywords_to_send in partial_overlap_queries:

                    # Get results using this query
                    returnedIDs = receiver.returnTuples([(signal, signal.keyword) for signal in keywords_to_send],
                                                        top_k)
                    rr = getRR(matchingIDs, returnedIDs)

                    # Keep track of first best query using all keywords
                    if best_rr[index] < rr:
                        best_rr[index] = rr
                        best_queries_filler_queries = []
                        best_queries_filler_queries.append((keywords_to_send, rr))

                    # Can't do better than 1
                    if best_rr[index] == 1:
                        break
                best_queries[tupID][index] += best_queries_filler_queries

        best_queries[tupID].append([rr for rr in best_rr])

    if sample_set is None:
        with open(
                '{save_path}/bestQ-{dataset}_all-k={top_k}-q={min_length}to{max_length}-{start_range}_{end_range}.pkl',
                'wb') as f:
            pickle.dump(best_queries, f)
    else:
        with open(
                f'{save_path}/bestQ-{dataset}_sample-{len(sample_set)}-k={top_k}-q={min_length}to{max_length}-{start_range}_{end_range}.pkl',
                'wb') as f:
            pickle.dump(best_queries, f)