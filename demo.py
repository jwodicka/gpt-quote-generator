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

def main(
        # Required
        source: ('Source file to load data from', 'positional'),

        # Flags
        autodownload: ('Automatically download model if needed?', 'flag'),
        finetune: ('Run a fine-tuning pass on the model?', 'flag'),
        resume: ('Resume training from a prior run?', 'flag'),

        # Options
        # There are multiple GPT-2 models that have been released. 124M is the smallest, and it gives more than adequate results.
        model_name: ('Name of the GPT-2 model to use', 'option')='124M',
        run_name: ('Name to give this run - used for resuming prior runs', 'option')='run1',
        steps: ('Number of steps of training to carry out', 'option', None, int)=100,
        nsamples: ('Number of generation passes to run', 'option', None, int)=1,
    ):

    model_directory = 'models' # If we want, we could make this configurable, but there's some testing involved to make
                               # sure we do so consistently everywhere.

    print(f"Using model: {model_name}")

    # Do we have the base model? If not, we need to get it.
    if not os.path.isdir(os.path.join(model_directory, model_name)):
        if autodownload:
            print(f"Downloading {model_name} model...")
            gpt2.download_gpt2(model_name=model_name, model_dir=model_directory)
        else:
            print(f"Couldn't find {model_name} in {model_directory}. Turn on autodownload to fetch it.")
            return


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

    # Original training was performed on 1000 steps.
    # At step 1000 we were frequently reiterating lines in the source.
    # At step 400 we were sometimes recycling existing quotes.

    if resume:
        print(f"Loading run {run_name}")
        gpt2.load_gpt2(sess, run_name=run_name)


    if finetune:
        print('Fine-tuning the model from training data')
        gpt2.finetune(sess, file_name, model_name=model_name, steps=steps, run_name=run_name)


    print('Generating quotes')
    quotes_with_delimiters = gpt2.generate(sess, nsamples=nsamples, return_as_list=True)
    generated_quotes = []

    print('Checking for plagiarism')
    for sample in quotes_with_delimiters:
        print("SAMPLE: [" + sample + "]")
        generated_quotes.extend(find_quotes(sample))

    results = []
    for quote in generated_quotes:
        print("QUOTE: [" + quote + "]")
        closest_match = process.extractOne(quote, source_quotes)
        # print(closest_match)
        result = {
            "quote": quote,
            "best_match": closest_match[0],
            "match_score": closest_match[1],
        }
        results.append(result)

    novel_quote_count = 0
    with open('generated_quotes.v3.csv', 'w', newline='') as out_file:
        quote_writer = csv.DictWriter(out_file, fieldnames=['quote', 'best_match', 'match_score'])
        quote_writer.writeheader()
        for row in results:
            quote_writer.writerow(row)
            # Print the novel ones to console for funsies
            if row['match_score'] < 90:
                print(row['quote'])
                novel_quote_count += 1

    print('Novel quotes generated: {} of {} ({:.2%})'.format(novel_quote_count, len(results), novel_quote_count/len(results)))

if __name__ == '__main__':
    import plac; plac.call(main)