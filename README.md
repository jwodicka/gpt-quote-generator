## GPT Quote Generator

This project was developed to support Loading Ready Run's desire to have better bot-generated quotes for use in some
comedy games. It uses the GPT-2 model trained and released by OpenAI as a basis for text generation, and fine-tunes
the model based on quotes from the extensive LRR quotes database.

### Getting started

NOTE: These instructions are tested on MacOS 10.15. They are likely to work more-or-less unchanged on other POSIX-y
      environments, and will probably require some finesse to get working on Windows.

Before you begin, you will need:
* An installation of Python 3.7.7
  * If you use `asdf` or a similar runtime tool, the `.tool-versions` file should specify the correct version. 
  * Other versions of Python may or may not work; if you test this with different Python versions feel free to report
    back about your results.

Installing dependencies:
1. Clone this repository to your computer.
2. Open a command-line at the root of the repository.
3. Create a virtual environment: `python -m venv venv`
4. Begin using the virtual environment: `source venv/bin/activate`
5. Install the needed dependencies: `pip install -r requirements.txt`
6. Choose a flavor of TensorFlow and install version 1.15:
   1. If you don't want to use your GPU, run `pip install tensorflow=1.15`
   2. If you want to use your GPU for processing, run `pip install tensorflow-gpu=1.15`
   * Currently, versions of TensorFlow later than 1.15 are not supported!


### Acknowledgments

This system is built on top of `gpt_2_simple`, which is doing most of the heavy lifting of actually working with the
GPT-2 model.

The `fuzzywuzzy` library is used for fuzzy-matching of text against the input database to detect cases where the model
has quoted the source material.
