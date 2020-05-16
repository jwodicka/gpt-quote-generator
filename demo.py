import gpt_2_simple as gpt2
import os
import requests
import csv
import re
from collections import Counter
from fuzzywuzzy import process

quote_matcher = re.compile('<\\|startoftext\\|>(.*?)<\\|endoftext\\|>')
def find_quotes(text):
    return quote_matcher.findall(text)

model_name = '124M'
# Do we have the base model? If not, we need to get it.
if not os.path.isdir(os.path.join('models', model_name)):
    print(f"Downloading {model_name} model...")
    gpt2.download_gpt2(model_name=model_name)

attributions = Counter()
# Track all individual quotes so that we can check whether we've regenerated them.
source_quotes = []
with open('quotes.20200503.csv', newline='') as quote_file:
    with open('barequotes.csv', 'w', newline='') as out_file:
        quote_reader = csv.DictReader(quote_file)
        quote_writer = csv.DictWriter(out_file, fieldnames=['quote'], extrasaction='ignore')
        quote_writer.writeheader() # Loader assumes there will be a header and skips it.
        for row in quote_reader:
            quote_writer.writerow(row)
            source_quotes.append(row['quote'])
            attributions[row['attrib_name']] += 1

print("Loaded {} quotes attributed to {} sources.".format(len(source_quotes), len(attributions)))
print("Top 10 sources:")
print(attributions.most_common(10))
# print(attributions)

file_name="barequotes.csv"

sess = gpt2.start_tf_sess()

steps = 600
# Original training was performed on 1000 steps.
# At step 1000 we were frequently reiterating lines in the source.
# At step 400 we were sometimes recycling existing quotes.

gpt2.finetune(sess, file_name, model_name=model_name, steps=steps, run_name='dump2020_05_03')
# gpt2.load_gpt2(sess)

print('Generating quotes')
quotes_with_delimiters = gpt2.generate(sess, nsamples=50, return_as_list=True)
generated_quotes = []

print('Checking for plagiarism')
for sample in quotes_with_delimiters:
    generated_quotes.extend(find_quotes(sample))

results = []
for quote in generated_quotes:
    # print("QUOTE: [" + quote + "]")
    closest_match = process.extractOne(quote, source_quotes)
    # print(closest_match)
    result = {
        "quote": quote,
        "best_match": closest_match[0],
        "match_score": closest_match[1], 
    }
    results.append(result)

novel_quote_count = 0
with open('generated_quotes.v2.csv', 'w', newline='') as out_file:
    quote_writer = csv.DictWriter(out_file, fieldnames=['quote', 'best_match', 'match_score'])
    quote_writer.writeheader()
    for row in results:
        quote_writer.writerow(row)
        # Print the novel ones to console for funsies
        if row['match_score'] < 90:
            print(row['quote'])
            novel_quote_count += 1

print('Novel quotes generated: {} of {} ({:.2%})'.format(novel_quote_count, len(results), novel_quote_count/len(results)))
