from importlib.metadata import distribution
import os
import random
import re
import sys
import numpy as np


DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    all_pgs = list(corpus.keys())
    num_of_all_pgs = len(all_pgs)
    pg_distribution = {}
    linked_to_pgs = corpus[page]
    num_of_linked_to_pgs = len(linked_to_pgs)
    # Two cases of choosing a page
    # 1. randomly choose one of all pages in the corpus with equal probability
    for pg in all_pgs:
        pg_distribution[pg] = (1 - damping_factor) / num_of_all_pgs
    # 2. randomly choose one of the links from page with equal probability`
    
    #If page has no outgoing links, 
    # then transition_model should return a probability distribution 
    # that chooses randomly among all pages with equal probability. 
    if num_of_linked_to_pgs > 0:
        for linked_to_pg in linked_to_pgs:
            pb = damping_factor / num_of_linked_to_pgs
            pg_distribution[linked_to_pg] += pb
    else:
        for pg in all_pgs:
            pg_distribution[pg] = 1 / num_of_all_pgs
    return pg_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pr_values = {}
    # starting with a page at random
    pg = random.choice(list(corpus.keys()))

    # sample n times
    for i in range(n):
        # pb distribution of pages that pg linked to
        pb_distribution = transition_model(corpus, pg, damping_factor)
        # find the new page according to p distribution of the current page
        pgs = list(pb_distribution.keys())
        probs = list(pb_distribution.values())
        pg = np.random.choice(pgs, p=probs)
        # increase the estimated value by 1 / n
        if pg in pr_values.keys():
            pr_values[pg] += 1 / n
        else:
            pr_values[pg] = 1 / n

    return pr_values


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold = 0.001
    all_pgs = list(corpus.keys())
    N = len(all_pgs)
    pr_distribution = {}
    pr_value_change = {}
    # NumLinks(i) is the number of links on page i.
    num_links = {}
    # assigning each page a rank of 1 / N
    # assign 1 as the initial value change of each page 
    for pg in all_pgs:
        pr_distribution[pg] = 1 / N
        pr_value_change[pg] = 1
        num_links[pg] = len(corpus[pg])
    # repeatedly calculate new rank values 
    # based on all of the current rank values
    value_change = True
    const_part = (1 - damping_factor) / N 
    while value_change:
        # print(pr_distribution)
        new_pr = {}
        # calculate new rank values
        for page in all_pgs:
            pr = 0
            # IMPORTANT 
            # For a corpus as follows
            # {'4.html': {'3.html'}, '3.html': {'4.html', '2.html'}, '2.html': {'1.html', '3.html'}, '1.html': {'2.html'}}
            # pages linked to the current page should be those in the value part
            # e.g. Pages linked to page4 is p3, not p2
            for pg in all_pgs:
                # A page that has no links at all should be interpreted as 
                # having one link for every page in the corpus (including itself)
                # e.g. {'4.html': set(), '3.html': {'4.html', '2.html'}, '2.html': {'1.html', '3.html'}, '1.html': {'2.html'}}
                if len(corpus[pg]) == 0:
                    pr += pr_distribution[pg] / N
                elif page in corpus[pg]:
                    pr += pr_distribution[pg] / num_links[pg]    
            
            pr = damping_factor * pr + const_part
            # find the ABSOLUTE value change 
            change = abs(pr - pr_distribution[page])
            pr_value_change[page] = change
            # store pr in new_pr
            new_pr[page] = pr
            
        # update pr
        pr_distribution = new_pr
        # detect if there is still value change greater than .001
        value_change = False
        for pg in all_pgs:
            if pr_value_change[pg] > threshold:
                value_change = True
                break
        
    return pr_distribution

if __name__ == "__main__":
    main()
