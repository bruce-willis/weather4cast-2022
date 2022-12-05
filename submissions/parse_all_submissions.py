import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

heldout_core_links = [
    'https://pastebin.com/VVwwxE8c',  # official baseline
    'https://pastebin.com/Zb5skMqm',  # baseline bce imporved 15
    'https://pastebin.com/tbym1PRZ',  # swin bce epoch 3
    'https://pastebin.com/J90bfciv',  # majority voting
]

heldout_transfer_links = [
    'https://pastebin.com/f5C1QH38',  # swin bce channel conv adabelief epoch 4
    'https://pastebin.com/2fAQZZhu',  # swin bce adabelief epoch 4
]


def get_test_core_links():
    # extract all pastebin links from readme file using regex
    readme = Path('submissions/readme.md').read_text()
    pastebin_links = re.findall(r'https://pastebin.com/.*\)', readme)
    pastebin_links.insert(0, 'https://pastebin.com/N0wnFMRi')  # official baseline
    return pastebin_links


def main():
    results = defaultdict(list)
    results_per_submission = defaultdict(lambda: defaultdict(list))
    total_results = dict()
    for link in get_test_core_links():  #get_test_core_links()|heldout_core_links:
        link = link.strip().rstrip('/').rstrip(')')
        link = link.replace('https://pastebin.com/', 'https://pastebin.com/raw/')
        # download raw text from url
        with urllib.request.urlopen(link) as response:
            raw_text = response.read().decode('utf-8')
        # extract name of sumbission starting from world 'Notes':
        if 'Notes:' in raw_text:
            name = raw_text.rpartition('Notes: ')[2].split('\n')[0].strip()
        else:
            name = 'vivit_bceLoss_8_8_768'
        # extract all values per region
        total_score, *regions_score = raw_text.split('Score:')[1].split('\n')
        total_results[name] = total_score.strip()
        for region_score in regions_score:
            region, score = region_score.split(':')
            results[region].append((float(score), name))
            region_name, year = region.rpartition(' ')[2], region.rpartition(' ')[0].rpartition(' ')[2]
            results_per_submission[name][f'{region_name}_{year}'] = float(score)

    import joblib
    joblib.dump(dict(results_per_submission), 'results_per_submission.joblib')

    # print best results for overall and per region
    print(*sorted(total_results.items(), key=lambda x: x[1], reverse=True)[:10], sep='\n', end='\n\n')
    best_scores_per_region = []
    for region, scores in results.items():
        print(region)
        best_scores_per_region.append(max(scores)[0])

        for score, name in sorted(scores, reverse=True)[:6]:
            print(f'{score:.3f} {name}')
        print()

    print(np.mean(best_scores_per_region))


if __name__ == '__main__':
    main()
