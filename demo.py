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
        skip_plagiarism: ('Skip checking results for plagiarism?', 'flag'),

        # Options
        # There are multiple GPT-2 models that have been released. 124M is the smallest, and it
        # gives more than adequate results.
        model_name: ('Name of the GPT-2 model to use', 'option')='124M',
        run_name: ('Name to give this run - used for resuming prior runs', 'option')='run1',
        steps: ('Number of steps of training to carry out', 'option', None, int)=100,
        nsamples: ('Number of generation passes to run', 'option', None, int)=1,
        save_every: ('Save a checkpoint every this many steps', 'option', None, int)=200,
        sample_every: ('Sample the output during training every this many steps', 'option', None, int)=100,
        restore_from: ('Checkpoint to resume from', 'option')='latest',
        output_file: ('Name of the csv file to write', 'option')=None,
        delimiter: ('Character that delimits columns in source', 'option')=',',
        quote_column: ('Label for the column with quotes', 'option')='quote',
        attribution_column: ('Label for the column with attributions', 'option')='attrib_name',
    ):

    model_directory = 'models' # If we want, we could make this configurable, but there's some
                               # testing involved to make sure we do so consistently everywhere.
    temporary_input_file = 'temp_input.csv' # The file containing cleaned data from the quotes

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
    with open(source, newline='') as quote_file:
        with open(temporary_input_file, 'w', newline='') as out_file:
            quote_reader = csv.DictReader(quote_file, delimiter=delimiter)
            quote_writer = csv.DictWriter(out_file, fieldnames=[quote_column], extrasaction='ignore')
            quote_writer.writeheader() # Loader assumes there will be a header and skips it.
            for row in quote_reader:
                quote_writer.writerow(row)
                source_quotes.append(row[quote_column])
                attributions[row[attribution_column]] += 1

    print("Loaded {} quotes attributed to {} sources.".format(len(source_quotes), len(attributions)))
    print("Top 10 sources:")
    print(attributions.most_common(10))

    sess = gpt2.start_tf_sess()

    if resume:
        print(f"Loading run {run_name}")
        # If the model name is set, the run name will be ignored.
        gpt2.load_gpt2(sess, run_name=run_name)


    if finetune:
        print('Fine-tuning the model from training data')
        gpt2.finetune(
            sess,
            temporary_input_file,
            model_name=model_name,
            steps=steps,
            run_name=run_name,
            save_every=save_every,
            sample_every=sample_every,
            restore_from=restore_from,
            max_checkpoints=100, # How many checkpoints to keep for each run
        )


    print('Generating quotes')
    quotes_with_delimiters = gpt2.generate(
        sess,
        run_name=run_name,
        nsamples=nsamples,
        return_as_list=True
    )

    print('Parsing quotes')
    generated_quotes = []
    for sample in quotes_with_delimiters:
        print("SAMPLE: [" + sample + "]")
        generated_quotes.extend(find_quotes(sample))

    results = []
    if skip_plagiarism:
        for quote in generated_quotes:
            print("QUOTE: [" + quote + "]")
            results.append({"quote": quote})
    else:
        print('Checking for plagiarism')
        novel_quote_count = 0
        for quote in generated_quotes:
            closest_match = process.extractOne(quote, source_quotes)
            print("QUOTE: [" + quote + "]")
            if closest_match[1] >= 90:
                # This is a bit too close.
                print("MATCH: [" + closest_match[0] + "]")
            else:
                novel_quote_count += 1
            result = {
                "quote": quote,
                "best_match": closest_match[0],
                "match_score": closest_match[1],
            }
            results.append(result)

        print('Novel quotes generated: {} of {} ({:.2%})'.format(novel_quote_count, len(results), novel_quote_count/len(results)))

    if output_file == None:
        output_file = run_name + '_' + source

    if skip_plagiarism:
        fieldnames=['quote']
    else:
        fieldnames=['quote', 'best_match', 'match_score']

    with open(output_file, 'w', newline='') as out_file:
        quote_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        quote_writer.writeheader()
        for row in results:
            quote_writer.writerow(row)

if __name__ == '__main__':
    import plac; plac.call(main)
