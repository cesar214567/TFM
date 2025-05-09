import sys

#function to print the amount of videos processed the datasets
def print_progress(count, max_count):
    # Percentage completion.
    pct_complete = count / max_count

    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()